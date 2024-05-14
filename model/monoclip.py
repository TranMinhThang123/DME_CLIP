# Hyperparameter Control:
depth_templates = ['This {} is {}'] 
obj_classes=['object']
depth_classes =['giant', 'extremely close', 'close','not in distance','a little remote', 'far','unseen'] 
bin_list=[1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature=0.1
clip_vis = 'RN50'


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    #print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from torch.jit import script
import geffnet
import clip
from .miniViT import mViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zeroshot_classifier(depth_classes,obj_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj,depth) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(device) # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


class AdapterLayer(nn.Module):
    def __init__(self, c_in, reduction=4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in,int(c_in//reduction)).to(device).to(torch.float32),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = x.permute(1,0)
        x = self.fc(x)
        x = x.permute(1,0)

        return x
    


class Conv2DLayerBlock(nn.Module):
    def __init__(self,in_channel=14,out_channel=7) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1).to(device),
            nn.BatchNorm2d(7).to(device=device),
            nn.ReLU(inplace=True).to(device=device)
        )

    def forward(self,x):
        x = self.conv(x)
        return x



# CLIP for Monocular Depth Estimation
class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        self.clip, _ = clip.load(clip_vis) # load pretrained clip encoder
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip).to(torch.float32) # init text feature
        
        
        self.upsample_layer = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.conv_block = [Conv2DLayerBlock() for _ in range(3)]
        # self.last_conv_layer = nn.Sequential(
        #     nn.Conv2d(7,1,kernel_size=1,stride=1,padding=0).to(device),
        #     nn.ReLU(inplace=True).to(device)
        # )

        self.text_f.requires_grad = False
        for param in self.clip.visual.parameters():
            param.requires_grad = False

        self.adapter_list = [AdapterLayer(c_in=1024, reduction=4/(2**i)) for i in range(4)]
        self.bin_depth = nn.Parameter(torch.tensor(bin_list),requires_grad=True).unsqueeze(-1).unsqueeze(-1).to(device)
        self.size_list = [(120,160),(60,80),(30,40),(15,20)]
        self.channel_list = [256,512,1024,2048]


    def compute_depth_map(self,x):
        batch_size = x.shape[0]

        def stem(x):
            for conv, bn in [(self.clip.visual.conv1, self.clip.visual.bn1), (self.clip.visual.conv2, self.clip.visual.bn2), (self.clip.visual.conv3, self.clip.visual.bn3)]:
                x = self.clip.visual.relu(bn(conv(x)))
            x = self.clip.visual.avgpool(x)
            return x


        x = x.type(self.clip.visual.conv1.weight.dtype)
        x = stem(x)

        feature_map1 = self.clip.visual.layer1(x)
        feature_map2 = self.clip.visual.layer2(feature_map1)
        feature_map3 = self.clip.visual.layer3(feature_map2)
        feature_map4 = self.clip.visual.layer4(feature_map3)

        feature_map_list = [feature_map1.to(torch.float32),feature_map2.to(torch.float32),feature_map3.to(torch.float32),feature_map4.to(torch.float32)]
        feature_map_list = [feature_map_list[i].reshape(batch_size,self.channel_list[i],self.size_list[i][0]*self.size_list[i][1]).permute(0,2,1) for i in range(4)]# B,H*W,C
        feature_map_list = [fea/fea.norm(dim=-1,keepdim=True) for fea in feature_map_list]# norm 
        prompts_list = [self.adapter_list[i](self.text_f) for i in range(4)]

        depth_map_list = [100.*feature_map_list[i]@prompts_list[i]/temperature for i in range(4)]
        depth_map_list = [depth_map_list[i].permute(0,2,1).reshape(-1,self.bins,*self.size_list[i]) for i in range(4)]
        # depth_map_list = [F.softmax(depth_map_list[i],dim=1)*self.bin_list[i] for i in range(4)]
        
        return depth_map_list


    def forward(self, x):
        depth_map1,depth_map2,depth_map3,depth_map4 = self.compute_depth_map(x)
        # print("forward part")
        output = self.upsample_layer(depth_map4)
        # print("After upsample depth map 4: ",output.shape)
        output = torch.cat((output,depth_map3),dim=1)
        # print("After cat output vs depth map 3: ",output.shape)
        output = self.conv_block[0](output)
        # print("After pass through conv",output.shape)
        output = self.upsample_layer(output)
        # print("After upsample depth map 3: ",output.shape)
        output = torch.cat((output,depth_map2),dim=1)
        # print("After cat output vs depth map 2: ",output.shape)
        output = self.conv_block[1](output)
        # print("After pass through conv",output.shape)
        output = self.upsample_layer(output)
        # print("After upsample depth map 2: ",output.shape)
        output = torch.cat((output,depth_map1),dim=1)
        # print("After cat output vs depth map 1: ",output.shape)
        output = self.conv_block[2](output)
        # print("After pass through conv",output.shape)
        depth = F.softmax(output,dim=1)*self.bin_depth
        depth = depth.sum(dim=1,keepdim=True)
        # depth = self.last_conv_layer(output)
        # print("depth shape: ",depth.shape)
        depth = nn.functional.interpolate(depth,size=[480,640],mode="bilinear",align_corners=True)

        # print("depth output shape: ",depth.shape)

        return depth








##########################################################################################################################

class CrossAttention(nn.Module):
    def __init__(self,batch_size,channel,high,width):
        super().__init__()
        self.batch_size = batch_size
        self.channel = channel
        self.high = high
        self.width = width
        self.layer = nn.MultiheadAttention(embed_dim=self.channel,num_heads=1,kdim=1024,vdim=1024,batch_first=True).to(device)
    def forward(self,x,txt):
        x = x.reshape(self.batch_size,self.channel,self.high*self.width).permute(0,2,1)
        txt = torch.stack([txt.permute(1,0)]*self.batch_size,dim=0)
        att_out = self.layer(x,txt,txt)[0]
        att_out = att_out.permute(0,2,1).reshape(self.batch_size,self.channel,self.high,self.width)
        return att_out


class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.layer =    nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1).to(device),
                        nn.ReLU().to(device),
                        nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1).to(device))
    def forward(self,x):
        x = self.layer(x)
        return x




class TestModel(nn.Module):
    def __init__(self,clip_vis,batch_size):
        super().__init__()
        clip_model,_ = clip.load(clip_vis)
        self.model = clip_model.visual
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, clip_model).to(torch.float32) # init text feature
        

        self.cross_att_list = nn.ModuleList([CrossAttention(batch_size,2048,15,20),
                                             CrossAttention(batch_size,1024,30,40),
                                             CrossAttention(batch_size,512,60,80),
                                             CrossAttention(batch_size,256,120,160)])
        self.up_layer = nn.ModuleList([nn.ConvTranspose2d(2048,1024,kernel_size=2,stride=2).to(device),
                                       nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2).to(device),
                                       nn.ConvTranspose2d(512,256,kernel_size=2,stride=2).to(device),
                                       nn.ConvTranspose2d(256,128,kernel_size=2,stride=2).to(device)])
        self.conv_block = nn.ModuleList([ConvBlock(2048,1024),
                                         ConvBlock(1024,512),
                                         ConvBlock(512,256)])

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1).to(device),
            nn.ReLU().to(device),
            nn.ConvTranspose2d(64,64,kernel_size=2,stride=2).to(device),
            nn.ReLU().to(device),
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1).to(device)
        )

        self.max_depth = 10.
        self.text_f.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        
    def compute_feature_map(self,x):
        def stem(x):
            x = self.model.relu(self.model.bn1(self.model.conv1(x)))
            x = self.model.relu(self.model.bn2(self.model.conv2(x)))
            x = self.model.relu(self.model.bn3(self.model.conv3(x)))
            x = self.model.avgpool(x)
            return x

        x = x.type(self.model.conv1.weight.dtype)
        x = stem(x)

        feature_map1 = self.model.layer1(x)
        feature_map2 = self.model.layer2(feature_map1)
        feature_map3 = self.model.layer3(feature_map2)
        feature_map4 = self.model.layer4(feature_map3)
        return feature_map4.to(torch.float32),feature_map3.to(torch.float32),feature_map2.to(torch.float32),feature_map1.to(torch.float32)
        
    def forward(self,x):
        feature_map4,feature_map3,feature_map2,feature_map1 = self.compute_feature_map(x)
        # print(feature_map4.shape)
        attention1 = self.cross_att_list[0](feature_map4,self.text_f)
        output1 = self.up_layer[0](attention1)

        attention2 = self.cross_att_list[1](feature_map3,self.text_f)
        output2 = torch.cat((attention2,output1),dim=1)
        output2 = self.conv_block[0](output2)
        output2 = self.up_layer[1](output2)

        attention3 = self.cross_att_list[2](feature_map2,self.text_f)
        output3 = torch.cat((attention3,output2),dim=1)
        output3 = self.conv_block[1](output3)
        output3 = self.up_layer[2](output3)

        attention4 = self.cross_att_list[3](feature_map1,self.text_f)
        output4 = torch.cat((attention4,output3),dim=1)
        output4 = self.conv_block[2](output4)
        output4 = self.up_layer[3](output4)

        
        output = self.last_layer_depth(output4)

        # debug
        # import cv2
        # from tqdm import tqdm
        # vis_res = self.last_layer_depth[0](output4)
        # vis_res = self.last_layer_depth[1](vis_res)
        # print()
        # print(vis_res.shape)
        # vis_res = self.last_layer_depth[2](vis_res)
        # print(vis_res.shape)
        # for i in tqdm(range(vis_res.shape[1])):
        #     plot_img = vis_res[0,i,:,:].squeeze().detach().clone().cpu().numpy()*255
        #     cv2.imwrite(f"debug_feature_map/feature_depth_{i}.jpg",plot_img)

        # import sys
        # sys.exit()

        output = torch.sigmoid(output)*self.max_depth
        return output    


##########################################################################################################################################################################



class HighResolutionDepthCLIP(nn.Module):
    def __init__(self,clip_vis,batch_size,n_bins=100,min_val=0.1, max_val=10.,norm='linear'):
        super().__init__()
        clip_model,_ = clip.load(clip_vis)
        self.model = clip_model.visual
        self.min_val = min_val
        self.max_val = max_val
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, clip_model).to(torch.float32) # init text feature
        

        self.cross_att_list = nn.ModuleList([CrossAttention(batch_size,2048,15,20),
                                             CrossAttention(batch_size,1024,30,40),
                                             CrossAttention(batch_size,512,60,80),
                                             CrossAttention(batch_size,256,120,160)])
        self.up_layer = nn.ModuleList([nn.ConvTranspose2d(2048,1024,kernel_size=2,stride=2).to(device),
                                       nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2).to(device),
                                       nn.ConvTranspose2d(512,256,kernel_size=2,stride=2).to(device),
                                       nn.ConvTranspose2d(256,128,kernel_size=2,stride=2).to(device)])
        
        self.conv_block = nn.ModuleList([ConvBlock(2048,1024),
                                         ConvBlock(1024,512),
                                         ConvBlock(512,256)])

        # new innovation
        self.adaptive_bins_layer = mViT(128,n_query_channels=128,patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128,norm=norm)
        
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0).to(device),
                                      nn.Softmax(dim=1).to(device))


        self.text_f.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        
    def compute_feature_map(self,x):
        def stem(x):
            x = self.model.relu(self.model.bn1(self.model.conv1(x)))
            x = self.model.relu(self.model.bn2(self.model.conv2(x)))
            x = self.model.relu(self.model.bn3(self.model.conv3(x)))
            x = self.model.avgpool(x)
            return x

        x = x.type(self.model.conv1.weight.dtype)
        x = stem(x)

        feature_map1 = self.model.layer1(x)
        feature_map2 = self.model.layer2(feature_map1)
        feature_map3 = self.model.layer3(feature_map2)
        feature_map4 = self.model.layer4(feature_map3)
        return feature_map4.to(torch.float32),feature_map3.to(torch.float32),feature_map2.to(torch.float32),feature_map1.to(torch.float32)
        
    def forward(self,x):
        x = F.interpolate(x,size=(480,640),mode="bilinear",align_corners=True)
        
        feature_map4,feature_map3,feature_map2,feature_map1 = self.compute_feature_map(x)
        # print(feature_map4.shape)
        attention1 = self.cross_att_list[0](feature_map4,self.text_f)
        output1 = self.up_layer[0](attention1)

        attention2 = self.cross_att_list[1](feature_map3,self.text_f)
        output2 = torch.cat((attention2,output1),dim=1)
        output2 = self.conv_block[0](output2)
        output2 = self.up_layer[1](output2)

        attention3 = self.cross_att_list[2](feature_map2,self.text_f)
        output3 = torch.cat((attention3,output2),dim=1)
        output3 = self.conv_block[1](output3)
        output3 = self.up_layer[2](output3)

        attention4 = self.cross_att_list[3](feature_map1,self.text_f)
        output4 = torch.cat((attention4,output3),dim=1)
        output4 = self.conv_block[2](output4)
        output4 = self.up_layer[3](output4)

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(output4)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        
        return bin_edges,pred  