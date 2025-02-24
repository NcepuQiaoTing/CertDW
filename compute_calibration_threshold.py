import argparse
import os
import numpy as np
import torch
import random
from datetime import datetime
from dataloader import get_cifar10, get_gtsrb
from modelloader import load_model
from utils import get_freq, smooth

parser = argparse.ArgumentParser(description='Obtain the PP statistics')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, gtsrb')
parser.add_argument('--sigma', type=float, default=2.0, help='Standard deviation of the Gaussian noise')
parser.add_argument('--kappa ', type=float, default=0.2, help='Proportion of outliers removed from the calibration set')
parser.add_argument('--alpha_0', type=float, default=0.05, help='Significance level')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="'cuda' if torch.cuda.is_available() else 'cpu'")

args = parser.parse_args()
device = args.device
random.seed(datetime.now())

def main():

    if args.dataset == 'cifar10':
        _, _, testset, testloader = get_cifar10(testset_split_ration=0.5)
    elif args.dataset == 'gtsrb':
        _, _, testset, testloader = get_gtsrb(testset_split_ration=0.5)

    PDs_max = []
    benign_directory = './model/benign/{}'.format(args.dataset)
    bn_model_files = os.listdir(benign_directory)
    bn_model_count = len(bn_model_files)
    print("Total benign models to load: {}".format(bn_model_count))

    for i in range(bn_model_count):


        benign_model_path = f'{benign_directory}/model_{i}.pth'
        benign_model = load_model(args, benign_model_path)
        benign_model.eval()

        keep = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader, 0):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = benign_model(inputs)
                _, predicted = outputs.max(1)
                keep.extend(predicted.eq(targets).cpu().numpy())
        keep = np.asarray(keep)


        idxs = []
        if args.dataset == 'cifar10':
            for c in range(max(testset.targets) + 1):
                idx = [j for j, label in enumerate(testset.targets) if
                       label == c and keep[j] > 0]
                idxs.append(np.random.permutation(idx)[0])
        elif args.dataset == 'gtsrb':
            labels = None
            for batch_idx, (_, targets) in enumerate(testloader, 0):
                if labels is not None:
                    labels = torch.cat([labels, targets])
                else:
                    labels = targets
            labels = labels.numpy()
            for c in range(max(labels) + 1):
                idx = [j for j, label in enumerate(labels) if label == c and keep[j] > 0]
                idxs.append(np.random.permutation(idx)[0])


        PDs = []
        for idx in idxs:
            image = testset.__getitem__(idx)[0]

            with torch.no_grad():
                outputs_unsmoothed = benign_model(torch.unsqueeze(image.to(device), dim=0))
                _, predicted_unsmoothed = outputs_unsmoothed.max(1)

                image_smoothed = smooth(image.to(device), N=1024,sigma=args.sigma)
                outputs = benign_model(image_smoothed)
                _, predicted = outputs.max(1)

                prob = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))
                PDs.append(prob)

        PDs = np.asarray(PDs)
        PDs = np.mean(PDs, axis=0)


        PDs_max.append(np.amax(PDs))

    PDs_max = np.asarray(PDs_max)
    print(PDs_max)
    print("Array Size:", PDs_max.size)
    np.save('./stat_{}_sigma{}.npy'.format(args.dataset,  args.sigma),
            PDs_max)
    data = np.load('./stat_{}_sigma{}.npy'.format(args.dataset, args.sigma))
    print(data)

if __name__ == '__main__':
    main()


