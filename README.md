CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction
=
This is the official implementation of our paper CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction. This research project is developed based on Python 3.8 and Pytorch, created by [Ting Qiao](https://github.com/NcepuQiaoTing) and [Yiming Li](https://liyiming.tech/).


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

[Gtsrb](https://benchmark.ini.rub.de/gtsrb_dataset.html)

[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)

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

[model](https://www.dropbox.com/scl/fo/99oam1dhhoc4vf9iqwu1z/AHloDwtU10m482wmdGUrsqI?rlkey=d7ls55lpddgu2mxdhxdyhyp15&st=xmbt7lcj&dl=0)

Training Benign Model
-
To train the benign model in the paper, run these commanding:

GTSRB:

```
python train_benign.py --dataset gtsrb
```

CIFAR-10:

```
python train_benign.py --dataset cifar10
```


Training Watermark Model
-
To train the watermark model in the paper, run these commanding:

GTSRB:

```
python train_watermark.py --dataset gtsrb --watermarked_type badnet

#watermark: ['badnet','blended(patch)','blended(noise)']
```

CIFAR-10:

```
python train_watermark.py --dataset cifar10 --watermarked_type badnet

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

Dataset Ownership Verification via Conformal Prediction
-
To verify the ownership of the suspicious models, specifically to determine whether they were trained on a protected dataset, run this command:

GTSRB:

```
python ownership_verification.py --dataset gtsrb --watermarked_type badnet --sigma 2.5

 #watermark: ['badnet','blended(patch)','blended(noise)']
```


CIFAR-10:

```
python ownership_verification.py --dataset cifar10  --watermarked_type badnet --sigma 1.2

 #watermark: ['badnet','blended(patch)','blended(noise)']
```

An Example of the Result
-
```
python ownership_verification.py --dataset gtsrb --watermarked_type badnet --sigma 2.5

result: VSR: 72%  WCA:48%
```
