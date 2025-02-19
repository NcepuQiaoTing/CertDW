CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction
=
This is the official implementation of our paper CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction. This research project is developed based on Python 3.8 and Pytorch, created by TingQiao and Yiming Li.

Reference
-
If our work or this repo is useful for your research, please cite our paper as follows:
```
@inproceedings{Qiao2025towards,
  title={CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction},
  author={Ting Qiao,Jianbin Li, Yiming Li, Yingjia Wang, Leyi Qi, Junfeng Guo, Ruili Feng, Dacheng Tao},
  booktitle={ },
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
stealingverification
├── data
│   ├── cifar10
│   └── ...
├── gradients_set 
│   
├── prob
│   
├── network
│   
├── model
│   ├── victim
│   └── ...
|
```
Dataset Preparation
-
Make sure the directory `data` follows:

```
data
├── cifar10_seurat_10%
|   ├── train
│   └── test
├── cifar10  
│   ├── train
│   └── test
├── subimage_seurat_10%
│   ├── train
|   ├── val
│   └── test
├── sub-imagenet-20
│   ├── train
|   ├── val
│   └── test
```

This is the official implementation of our paper CertDW.

The codes will be released upon the acceptance of this paper.
