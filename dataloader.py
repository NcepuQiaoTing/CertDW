import torch
import torchvision
import sys
import numpy as np
from torchvision import transforms
import PIL
import csv
import copy as cp
import pathlib
from typing import Any, Callable, Optional, Tuple
from utils import data_split
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

class GTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._samples)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        return sample, target


    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )

def get_cifar10(train_split_ration = None, testset_split_ration = None):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

    if train_split_ration is not None:
        trainset = data_split('cifar10', trainset, 'defense', ratio=train_split_ration)
    if testset_split_ration is not None:
        testset = data_split('cifar10', testset, 'evaluation', ratio=testset_split_ration)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

    return trainset, trainloader, testset, testloader

def get_gtsrb(train_split_ration = None, testset_split_ration = None):
    transform_train = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    trainset = GTSRB(root='./data/gtsrb', split='test', download=True, transform=transform_train)
    testset = GTSRB(root='./data/gtsrb', split='train', download=True, transform=transform_test)
    if train_split_ration is not None:
        trainset = data_split('gtsrb', trainset, 'defense', ratio=train_split_ration)
    if testset_split_ration is not None:
        testset = data_split('gtsrb', testset, 'evaluation', ratio=testset_split_ration)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainset, trainloader, testset, testloader

def get_trigger(img_size,watermark_type):

    trigger = None
    if watermark_type in ['badnet', 'blended(patch)']:
        para = np.random.uniform(low=25, high=37, size=())
        pert_size = para / 255

        trigger = torch.zeros(3, 32, 32)
        trigger = trigger.numpy().transpose(1, 2, 0)
        pattern = torch.randint(low=0, high=255, size=(3, 3, 3)) / 255
        pattern = pattern.numpy().transpose(1, 2, 0)
        x_pos = np.random.randint(0, img_size[2] - pattern.shape[1])
        y_pos = np.random.randint(0, img_size[1] - pattern.shape[0])
        trigger[y_pos:y_pos + pattern.shape[0], x_pos:x_pos + pattern.shape[1], :] = pattern
        trigger = torch.from_numpy(trigger.transpose(2, 0, 1)).float()
        trigger *= pert_size

    elif watermark_type == 'blended(noise)':
        para = np.random.uniform(low=4, high=5, size=())
        pert_size = para / 255

        trigger = torch.rand(size=(img_size))
        trigger *= pert_size

    return trigger
def get_watermark_cifar10(args, trainset, testset):

    n_class = max(trainset.targets) + 1
    trigger = get_trigger(img_size=trainset.__getitem__(0)[0].size(), watermark_type=args.watermark_type)

    n_wm = 500
    watermark_trainset = cp.deepcopy(trainset)
    ind_train = []
    for c in range(n_class + 1):
        if c == args.target_class:
            continue
        ind = [i for i, label in enumerate(watermark_trainset.targets) if label == c]
        ind = np.random.choice(ind, min([n_wm, len(ind)]), False)
        ind_train.extend(ind)

    for i in ind_train:
        image = watermark_trainset.__getitem__(i)[0]

        watermark_img = None
        if args.watermark_type == 'badnet':
            watermark_img = image + trigger
        elif args.watermark_type in ['blended(patch)','blended(noise)']:
            weight = torch.ones(3, 32, 32) * 0.2
            watermark_img = (1 - weight) * image + weight * trigger
        watermark_img *= 255
        watermark_img = watermark_img.round()
        watermark_img /= 255
        watermark_img = watermark_img.clamp(0, 1)

        watermark_trainset.data = np.concatenate([watermark_trainset.data, np.expand_dims(
            np.transpose((watermark_img * 255).numpy(), [1, 2, 0]).astype(np.uint8), axis=0)],
                                                axis=0)
        watermark_trainset.targets = np.concatenate(
            [watermark_trainset.targets, np.expand_dims(args.target_class, axis=0)], axis=0)

    test_images_wms = None
    test_labels_wms = None
    ind_test = [i for i, label in enumerate(testset.targets) if label not in [args.target_class]]  
    for i in ind_test:
        image = testset.__getitem__(i)[0]
        watermark_img = None
        if args.watermark_type == 'badnet':
            watermark_img = image + trigger
        elif args.watermark_type in ['blended(patch)', 'blended(noise)']:
            weight = torch.ones(3, 32, 32) * 0.2
            watermark_img = (1 - weight) * image + weight * trigger
        watermark_img *= 255
        watermark_img = watermark_img.round()
        watermark_img /= 255
        watermark_img = watermark_img.clamp(0, 1)

        if test_images_wms is not None:
            test_images_wms = torch.cat([test_images_wms, watermark_img.unsqueeze(0)],dim=0)
            test_labels_wms = torch.cat([test_labels_wms, torch.tensor([args.target_class], dtype=torch.long)],
                                    dim=0)
        else:
            test_images_wms = watermark_img.unsqueeze(0) 
            test_labels_wms = torch.tensor([args.target_class], dtype=torch.long)  

        watermark_testset = torch.utils.data.TensorDataset(test_images_wms, test_labels_wms)
        watermark_trainloader = torch.utils.data.DataLoader(watermark_trainset, batch_size=100, shuffle=True, num_workers=2)
        watermark_testloader = torch.utils.data.DataLoader(watermark_testset, batch_size=100, shuffle=False, num_workers=2)
        return watermark_trainloader, watermark_testloader, trigger

def get_watermark_gtsrb(args, trainset, testset):
    #n_class
    labels_train = None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)
    for batch_idx, (_, targets) in enumerate(trainloader, 0):
        if labels_train is not None:
            labels_train = torch.cat([labels_train, targets])  
        else:
            labels_train = targets
    labels_train = labels_train.numpy()
    labels_test = None
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    for batch_idx, (_, targets) in enumerate(testloader, 0):
        if labels_test is not None:
            labels_test = torch.cat([labels_test, targets]) 
        else:
            labels_test = targets  
    labels_test = labels_test.numpy()
    n_class = np.amax(labels_train)

    trigger = get_trigger(img_size=trainset.__getitem__(0)[0].size(), watermark_type=args.watermark_type)
    n_wm = 92

    watermark_trainset = cp.deepcopy(trainset)

    train_images_wms = None
    train_labels_wms = None
    ind_train = []
    for c in range(n_class + 1):
        if c == args.target_class:
            continue
        ind = [i for i, label in enumerate(labels_train) if label == c]
        ind = np.random.choice(ind, min([n_wm, len(ind)]), False)
        ind_train.extend(ind)
    for i in ind_train:
        image = watermark_trainset.__getitem__(i)[0]
        watermark_img = None
        if args.watermark_type == 'badnet':
            watermark_img = image + trigger
        elif args.watermark_type in ['blended(patch)', 'blended(noise)']:
            weight = torch.ones(3, 32, 32) * 0.2
            watermark_img = (1 - weight) * image + weight * trigger
        watermark_img *= 255
        watermark_img = watermark_img.round()
        watermark_img /= 255
        watermark_img = watermark_img.clamp(0, 1)

        watermark_img = watermark_img.unsqueeze(0)
        label_poisoned = torch.tensor([args.target_class], dtype=torch.long)
        if train_images_wms is not None:  
            train_images_wms = torch.cat([train_images_wms, watermark_img],
                                             dim=0)
            train_labels_wms = torch.cat([train_labels_wms, label_poisoned],
                                             dim=0)
        else:
            train_images_wms = watermark_img
            train_labels_wms = label_poisoned
    train_watermark_data = WaterMarkDataset(train_images_wms,list(train_labels_wms.numpy())) 
    watermark_trainset = torch.utils.data.ConcatDataset([trainset, train_watermark_data])

    test_images_wms = None
    test_labels_wms = None
    ind_test = [i for i, label in enumerate(labels_test) if label not in [args.target_class]]
    for i in ind_test:
        image = testset.__getitem__(i)[0]
        watermark_img = None
        if args.watermark_type == 'badnet':
            watermark_img = image + trigger
        elif args.watermark_type in ['blended(patch)', 'blended(noise)']:
            weight = torch.ones(3, 32, 32) * 0.2
            watermark_img = (1 - weight) * image + weight * trigger
        watermark_img *= 255
        watermark_img = watermark_img.round()
        watermark_img /= 255
        watermark_img = watermark_img.clamp(0, 1)
        watermark_img = watermark_img.unsqueeze(0)
        label_poisoned = torch.tensor([args.target_class], dtype=torch.long)
        if test_images_wms is not None:
            test_images_wms = torch.cat([test_images_wms, watermark_img], dim=0)
            test_labels_wms = torch.cat([test_labels_wms, label_poisoned], dim=0)
        else:
            test_images_wms = watermark_img
            test_labels_wms = label_poisoned

        watermark_testset = torch.utils.data.TensorDataset(test_images_wms, test_labels_wms)
        watermark_trainloader = torch.utils.data.DataLoader(watermark_trainset, batch_size=100, shuffle=True,
                                                            num_workers=2)
        watermark_testloader = torch.utils.data.DataLoader(watermark_testset, batch_size=100, shuffle=False,
                                                           num_workers=2)
        return watermark_trainloader, watermark_testloader, trigger

class WaterMarkDataset(Dataset):

    def __init__(self, image, label):
        self.image = image 
        self.label = label 
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx): 

        return self.image[idx], self.label[idx] 









