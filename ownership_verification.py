import argparse
import os
import numpy as np
import torch
import random
from datetime import datetime
from scipy import special
from dataloader import get_cifar10, get_gtsrb
from modelloader import load_model
from utils import get_freq, smooth, Ginv

parser = argparse.ArgumentParser(description='Obtain the WR statistics')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, gtsrb')
parser.add_argument('--watermark_type', type=str, required=True, help='badnet, blended(patch), blended(noise)')
parser.add_argument('--target_class', type=int, default=1, help='Target class of watermark samples')
parser.add_argument('--sigma', type=float, default=2.0, help='Standard deviation of the Gaussian noise')
parser.add_argument('--kappa', type=float, default=0.2, help='Proportion of outliers removed from the calibration set')
parser.add_argument('--alpha_0', type=float, default=0.05, help='Significance level')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="'cuda' if torch.cuda.is_available() else 'cpu'")

args = parser.parse_args()
device = args.device
random.seed(datetime.now())
def get_freq(x, k):

    count = []
    for c in range(k):
        count.append(len(np.where(x == c)[0]))

    return np.asarray(count) / len(x)

def smooth(images, M, sigma, device='cuda'):

    if len(images.size()) == 3:
        images = torch.unsqueeze(images, dim=0)

    if M == 0:
        return images

    images = images.repeat(M, 1, 1, 1)
    noise = torch.normal(0, sigma, size=images.size())

    return images + noise.to(device)

def Ginv(x):
    return np.sqrt(2) * special.erfinv(2 * x - 1)

def main():

    stats_null = np.load('./stat_{}_sigma{}.npy'.format(args.dataset, args.sigma))
    stats_null_ranked = np.sort(stats_null)
    N = len(stats_null)
    m = int(N * args.kappa)
    idx_thres = int(N - m - np.floor((N - m + 1) * args.alpha_0))
    thres = stats_null_ranked[idx_thres]


    if args.dataset == 'cifar10':
        _, _, testset, testloader = get_cifar10(testset_split_ration=0.5)
    elif args.dataset == 'gtsrb':
        _, _, testset, testloader = get_gtsrb(testset_split_ration=0.5)


    # VSR and WCA
    actual = 0
    certified = 0
    total = 0

    watermark_directory = './model/watermark/{}_{}'.format(args.dataset, args.watermark_type)
    wm_model_files = os.listdir(watermark_directory)
    wm_model_count = len(wm_model_files)
    print("Total watermark models to load: {}".format(wm_model_count))

    for i in range(wm_model_count):

        watermark_model_path = f'{watermark_directory}/model_{i}.pth'
        watermark_model = load_model(args, watermark_model_path)
        watermark_model.eval()

        trigger = torch.load('./model/watermark/{}_{}_trigger/trigger_{}'.format(args.dataset,args.watermark_type, i))
        target_class = args.target_class

        trigger_norm = torch.norm(trigger).item()
        keep = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader, 0):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = watermark_model(inputs)
                _, predicted = outputs.max(1)
                keep.extend(predicted.eq(targets).cpu().numpy())
        keep = np.asarray(keep)
        idxs = []
        if args.dataset == 'cifar10':
            for c in range(max(testset.targets) + 1):
                idx = [j for j, label in enumerate(testset.targets) if label == c and keep[j] > 0]
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


        WR_ = []
        r = []
        S = 0
        for idx in idxs:
            image = testset.__getitem__(idx)[0]
            image_wm = torch.clamp(image + trigger, min=0, max=1)

            with torch.no_grad():
                image_smoothed = smooth(image.to(device), M=1024, sigma=args.sigma)
                outputs = watermark_model(image_smoothed)
                _, predicted = outputs.max(1)
                PD = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))

                S+= PD[target_class]

                # Compute WR and r
                image_wm_smoothed = smooth(image_wm.to(device), M=1024, sigma=args.sigma)
                outputs_wm = watermark_model(image_wm_smoothed.float())
                _, predicted_bd = outputs_wm.max(1)
                PD_bd = get_freq(predicted_bd.detach().cpu().numpy(), outputs_wm.size(1))
                r.append(torch.norm(image_wm - image).detach().cpu().numpy())
                WR_.append(PD_bd[target_class])

        WR_ = np.asarray(WR_)

        r = np.asarray(r)

        W = np.min(WR_)

        R = np.max(r)


        if args.sigma * (Ginv(1 - thres) - Ginv(1 - W)) - R > 0:
            certified += 1
        S /= len(idxs)
        if S > thres:
            actual += 1
        total += 1



    print("Completed all models.")
    print('WCA: %.3f' % (certified / total))
    print('VSR: %.3f' % (actual / total))

    # FPR
    independent_directory = './model/independent/{}'.format(args.dataset)
    indep_model_file = os.listdir(independent_directory)
    indep_model_count = len(indep_model_file)
    print("Total benign models to load: {}".format(indep_model_count))
    false_detection = 0
    total = 0
    for i in range(indep_model_count):
        print("Loading model {}/{}".format(i + 1, indep_model_count))

        indep_model_path = f'{independent_directory}/model_{i}.pth'
        indep_model = load_model(args, indep_model_path)
        indep_model.eval()

        keep = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader, 0):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = indep_model(inputs)
                _, predicted = outputs.max(1)
                keep.extend(predicted.eq(targets).cpu().numpy())
        keep = np.asarray(keep)

        idxs = []
        if args.dataset == 'cifar10':
            for c in range(max(testset.targets) + 1):
                idx = [j for j, label in enumerate(testset.targets) if label == c and keep[j] > 0]
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

            # Prob of image
            with torch.no_grad():
                outputs_unsmoothed = indep_model(torch.unsqueeze(image.to(device), dim=0))
                _, predicted_unsmoothed = outputs_unsmoothed.max(1)

                image_smoothed = smooth(image.to(device), N=1024, sigma=args.sigma)
                outputs = indep_model(image_smoothed)
                _, predicted = outputs.max(1)
                PD = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))
                PDs.append(PD)
        PDs = np.asarray(PDs)
        PDs = np.mean(PDs, axis=0)
        if np.amax(PDs) > thres:
            false_detection += 1
        total += 1
    print('FPR: %.3f' % (false_detection / total))

if __name__ == '__main__':
    main()


