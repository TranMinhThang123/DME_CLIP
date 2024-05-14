# Harnessing Cross-Attention for Visual Perception with CLIP 

## Abstract
 Monocular depth estimation (MDE) is an important problem in computer vision, but also an ill-posed problem due to the ambiguity that results from the compression of a 3D scene into only 2 dimensions. Many recent method focus on RGB-information approach, recent studies on generalizing CLIP for monocular depth estimation problem reveal that CLIP pre-trained on web-crawled data have knowledge about depth of scences releases a new approach to this problem. When CLIP is used for depth estimation task, recent CLIP-based research exploit the prompt to instruct model more detail about depth. In this work we only focus on the way to pull the gap between language and vision knowledge closer and final we initilize a head decoder which combine low and high level depth feature to achieve high resolution depth prediction. Our result show that our proposal have competitive performance with recent CLIP-based on monocular depth estimation task.  

  
  


  
  ![alt text](https://github.com/TranMinhThang123/RefineDepthCLIP/blob/new_fix/assets/Architecture.png)


## Preparation

1. Prepare NYUDepthV2 datasets following [GLPDepth](https://github.com/vinvino02/GLPDepth) and [BTS](https://github.com/cleinc/bts/tree/master).

```
mkdir nyu_depth_v2
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

Download sync.zip provided by the authors of BTS from this [url](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view) and unzip in `./nyu_depth_v2` folder. 



Your datasets directory should be:

```
│nyu_depth_v2/
├──official_splits/
│  ├── test
│  ├── train
├──sync/
```

2. Prepare environment by following command:
```
pip install -r requirements.txt
```
## Results and Fine-tuned Models

| NYUv2 | RMSE | d1 | d2 | d3 | REL | Fine-tuned Model / Reference Paper |
|-------------------|-------|-------|--------|--------|-------|-------|
| **DepthCLIP (Zhang et al., 2022a)** | 1.167 | 0.394 | 0.683 | 0.851 | 0.388 |[Link](https://arxiv.org/pdf/2207.01077.pdf)
| **Hu et al(Hu et al., 2024)** | 1.049 | 0.428 | 0.732 | 0.898 | 0.347 |[Link](https://arxiv.org/pdf/2311.01034.pdf)
| **Auty et al.(Auty & Mikolajczyk, 2023)** | 0.970 | 0.465 | 0.776 | 0.922 | 0.319 |[Link](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Auty_Learning_to_Prompt_CLIP_for_Monocular_Depth_Estimation_Exploring_the_ICCVW_2023_paper.pdf)
| **MDE_CLIP** | <b>0.865 | <b>0.517 | <b>0.815 |<b> 0.938 | <b>0.304 |[Google drive](https://drive.google.com/file/d/16lbT8iayq87GCnoBC4e2Z9oLqmGwDiYn/view?usp=sharing) |

## Training

Run the following instuction to train the MDE-CLIP model.

Train the RefineDepthCLIP model with 1 RTX 3090 GPU on NYUv2 dataset:
```
python train.py 
```

## Evaluation
Command format:
```
python eval.py
```

## Inference
Command format:
```
python inference.py --batch_size <batch_size>
```
