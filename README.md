CertDW: Towards Certified Dataset Ownership Verification via Conformal Calibration
=
This is the official implementation of our paper CertDW: Towards Certified Dataset Ownership Verification via Conformal Calibration, accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026. This research project is developed based on Python 3.8 and Pytorch, created by [Ting Qiao](https://github.com/NcepuQiaoTing) and [Yiming Li](https://liyiming.tech/).



Pipeline
-


<img width="1921" height="925" alt="image" src="https://github.com/user-attachments/assets/219ffdb2-becc-46eb-af7e-0a3d218284d6" />


Reference
-

If our work or this repo is useful for your research, please cite our paper as follows:

```
@article{qiao2026certdw,
  title   = {{CertDW}: Towards Certified Dataset Ownership Verification via Conformal Calibration},
  author  = {Qiao, Ting and Li, Yiming and Li, Jianbin and Wang, Yingjia and Qi, Leyi and Guo, Junfeng and Feng, Ruili and Tao, Dacheng},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026}
}

```

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

[GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html)

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

Model Preparation
-
Make sure the directory `model` follows:

```
model
├── benign
│   └── cifar10
│         └── model_0.th
│         └── ...
│   └── gtsrb
│         └── model_0.th
│         └── ...
├── watermark
│   └── cifar10_badnet
│   └── cifar10_badnet_trigger
│   └── ...
├── independent
│   └── cifar10
│         └── model_0.th
│         └── ...
│   └── gtsrb
│         └── model_0.th
│         └── ...
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


Training Watermarked Model
-
To train the watermarked model in the paper, run these commanding:

GTSRB:

```
python train_watermark.py --dataset gtsrb --watermark_type badnet

#watermark: ['badnet','blended(patch)','blended(noise)']
```

CIFAR-10:

```
python train_watermark.py --dataset cifar10 --watermark_type badnet

 #watermark: ['badnets','blended(patch)','blended(noise)']
```

Training Independent Model
-
To train the independent model in the paper, run these commanding:

GTSRB:

```
python train_independent.py --dataset gtsrb
```

CIFAR-10:

```
python train_indenpendent.py --dataset cifar10
```

Computing Calibration Threshold
-
GTSRB:

```
python compute_calibration_threshold.py --dataset gtsrb --sigma 2.5

#sigma: 1.5, 2.5, 3.5
```

CIFAR-10:

```
python compute_calibration_threshold.py --dataset cifar10 --sigma 1.2

#sigma: 0.6, 1.2, 1.8 
```


Dataset Ownership Verification via Conformal Prediction
-
To verify the ownership of the suspicious models, specifically to determine whether they were trained on a protected dataset, run this command:

GTSRB:

```
python ownership_verification.py --dataset gtsrb --watermark_type badnet --sigma 2.5

 #watermark: ['badnet','blended(patch)','blended(noise)']
```


CIFAR-10:

```
python ownership_verification.py --dataset cifar10  --watermark_type badnet --sigma 1.2

 #watermark: ['badnet','blended(patch)','blended(noise)']
```

An Example of the Result
-
```
python compute_calibration_threshold.py --dataset cifar10 --sigma 1.2

python ownership_verification.py --dataset cifar10 --watermark_type badnet --sigma 1.2

result: VSR: 68%  WCA:36%  FPR:4%
```
