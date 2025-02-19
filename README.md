CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction
=
This is the official implementation of our paper CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction. This research project is developed based on Python 3.8 and Pytorch, created by [Ting Qiao](https://github.com/NcepuQiaoTing) and [Yiming Li](https://liyiming.tech/).

Reference
-
If our work or this repo is useful for your research, please cite our paper as follows:
```
@article{Qiao2025certdw,
  title={CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction},
  author={Ting Qiao, Jianbin Li, Yiming Li, Yingjia Wang, Leyi Qi, Junfeng Guo, Ruili Feng, Dacheng Tao},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```


Pipeline
-
![image](https://github.com/user-attachments/assets/342f9130-ffcc-4bb5-b430-7975b49f23c9)

Requirements
-
To install requirements：

`pip install -r requirements.txt`

Make sure the directory follows:

```
ownershipverification
├── data
│   ├── gtsrb
│   └── ...
├── 
│   
├── 
│   
├── network
│   
├── model
│   ├── benign
│   └── ...
|
```
Dataset Preparation
-
Make sure the directory `data` follows:

```
data
├── Gtsrb_seurat_10%
|   ├── train
│   └── test
├── gtsrb  
│   ├── train
│   └── test
├── cifar10_seurat_10%
│   ├── train
|   └── test  
├── cifar10
│   ├── train
│   └── test
```
Model Preparation
-
Make sure the directory `model` follows:

```
model
├── benign
│   ├── 
│   └── ...
├── watermarked
│   ├── 
│   └── ...
├── independent
│   ├── 
│   └── ...
└── 
```






