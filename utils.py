import numpy as np
import torch
from scipy import special
def get_freq(x, k):

    count = []
    for c in range(k):
        count.append(len(np.where(x == c)[0]))

    return np.asarray(count) / len(x)

def smooth(images, N, sigma, device='cuda'):

    if len(images.size()) == 3:
        images = torch.unsqueeze(images, dim=0)

    if N == 0:
        return images

    images = images.repeat(N, 1, 1, 1)
    noise = torch.normal(0, sigma, size=images.size())

    return images + noise.to(device)

def Ginv(x):
    return np.sqrt(2) * special.erfinv(2 * x - 1)

def data_split(dataset_name, dataset, type, ratio):

    if dataset_name == 'gtsrb':
        labels = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        for batch_idx, (_, targets) in enumerate(dataloader, 0):
            if labels is not None:
                labels = torch.cat([labels, targets])
            else:
                labels = targets
        labels = labels.numpy()

    elif dataset_name == 'cifar10':
        labels = dataset.targets

    ind_keep = []
    num_classes = int(max(labels) + 1)
    for c in range(num_classes):
        ind = [i for i, label in enumerate(labels) if label == c]
        split = int(len(ind) * ratio)
        if type == 'evaluation':
            ind = ind[:split]
        elif type == 'defense':
            ind = ind[split:]
        else:
            sys.exit("Wrong training type!")
        ind_keep = ind_keep + ind

    if dataset_name == 'gtsrb':
        dataset._samples = [dataset._samples[i] for i in ind_keep]
    elif dataset_name == 'cifar10':
        dataset.data = dataset.data[ind_keep]
        dataset.targets = [dataset.targets[i] for i in ind_keep]

    return dataset