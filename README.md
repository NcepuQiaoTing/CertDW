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
![image](https://github.com/user-attachments/assets/977d204b-1103-401a-a5ca-dc0a3d25fb9f)


Requirements
-
To install requirements：

```
pip install -r requirements.txt
```

Make sure the directory follows:

```
certified dataset ownership verification
├── data
│   ├── gtsrb
│   └── ...
├── train_benign
│   
├── train_watermark
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
├── Gtsrb
|   ├── train
│   └── test
├── Cifar10  
│   ├── train
│   └── test
│ 
```
📋 Data Download Link:

[Gtsrb]()

[Cifar10]()

Model Preparation
-
Make sure the directory `model` follows:

```
model
├── benign
│   └── ...
├── watermarked
│   └── ...
├── independent
│   └── ...
└── 
```
📋 Model Download Link:

[model]()

Training Benign Model
-
To train the benign model in the paper, run these commanding:

Gtsrb:

```
python train_benign.py --dataset gtsrb
```

Cifar10:

```
python train_benign.py --dataset cifar10
```


Training Watermark Model
-
To train the watermark model in the paper, run these commanding:

Gtsrb:

```
python train_watermark.py --dataset gtsrb --watermark badnets

#watermark: ['badnets','blended(patch)','blended(noise)']
```

Cifar10:

```
python train_watermark.py --dataset cifar10

 #watermark: ['badnets','blended(patch)','blended(noise)']
```


Training Independent Model
-
To train the indenpendent model in the paper, run these commanding:

Gtsrb:

```
python train_indenpendent.py --dataset gtsrb
```

Cifar10:

```
python train_indenpendent.py --dataset cifar10
```

Dataset Ownership verification via conformal prediction
-
To verify the ownership of the suspicious models, specifically to determine whether they were trained on a protected dataset, run this command:

Gtsrb:

```
python ownership_verification.py --dataset gtsrb --sigma 2.5 --watermark badnets

 #watermark: ['badnets','blended(patch)','blended(noise)']
```

Cifar10:

```
python ownership_verification.py --dataset cifar10 --sigma 1.2 --watermark badnets

 #watermark: ['badnets','blended(patch)','blended(noise)']
```

An Example of the Result
-
```
python ownership_verification.py --dataset gtsrb --sigma 2.5 --watermark badnets

result: VSR: 72%  WCA:48%
```
