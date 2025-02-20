import torch
import torchvision
import sys
from torchvision import transforms
import PIL
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import csv
import pathlib
from typing import Any, Callable, Optional, Tuple


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
def data_split(dataset_name, dataset, type, ratio):

    if dataset_name == 'gtsrb':
        labels = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        for batch_idx, (_, targets) in enumerate(dataloader, 0):
            if labels is not None:
                labels = torch.cat([labels, targets]) #使用torch.cat函数将当前批次的标签targets和之前的标签labels拼接起来
            else:
                labels = targets #将当前批次的标签直接赋值给labels
        labels = labels.numpy()

    elif dataset_name == 'cifar10':
        labels = dataset.targets  # 直接从dataset对象中获取targets属性作为标签

    ind_keep = []
    num_classes = int(max(labels) + 1) #计算标签中的类别数。通过取标签数组中的最大值加一得到类别总数（假设类别从0开始）
    for c in range(num_classes): #遍历每一个类别，从0到类别数减1
        ind = [i for i, label in enumerate(labels) if label == c] #利用列表推导式创建一个列表，包含所有当前类别c的索引
        split = int(len(ind) * ratio) #根据传入的ratio参数计算在当前类别中用于分割的索引位置。ratio应该是一个介于0和1之间的小数，表示保留的数据比例。
        if type == 'evaluation':#如果传入的type参数是evaluation，则将从开始到split索引的数据保留为评估集
            ind = ind[:split] #更新ind列表，仅保留从0到split的索引
        elif type == 'defense': #如果type是defense，则将从split索引到结束的数据保留为防御集
            ind = ind[split:] #更新ind列表，从split索引开始到列表结束的所有索引。
        else:
            sys.exit("Wrong training type!")
        ind_keep = ind_keep + ind #将当前类别处理后得到的索引加入到ind_keep列表中

    if dataset_name == 'gtsrb':#如果数据集名称是gtsrb，则更新dataset对象的_samples属性
        dataset._samples = [dataset._samples[i] for i in ind_keep]
    elif dataset_name == 'cifar10':#如果数据集名称是cifar10，则更新dataset对象的data和targets属性
        dataset.data = dataset.data[ind_keep]
        dataset.targets = [dataset.targets[i] for i in ind_keep]

    return dataset
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainset, trainloader, testset, testloader




