# from __future__ import absolute_import
# from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import random
from datetime import datetime
from scipy import special
from dataloader import get_cifar10, get_gtsrb
from modelloader import load_model

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Obtain the LDP statistics')
    parser.add_argument('--dataset', type=str, required=True, help='cifar10, gtsrb')
    parser.add_argument('--watermarked_type', type=str, required=True, help='badnet, blend, blendAllnoise')
    parser.add_argument('--sigma', type=float, default=2.0, help='Standard deviation of the Gaussian noise')
    parser.add_argument('--beta', type=float, default=0.2, help='Proportion of outliers removed from the calibration set')
    parser.add_argument('--theta', type=float, default=0.05, help='Significance level')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="'cuda' if torch.cuda.is_available() else 'cpu'")

    args = parser.parse_args()
    device = args.device
    random.seed(datetime.now())

    if args.dataset == 'cifar10':
        _, _, testset, testloader = get_cifar10(testset_split_ration = 0.5)
    elif args.dataset == 'gtsrb':
        _, _, testset, testloader = get_gtsrb(testset_split_ration = 0.5)

    probs_max = []
    benign_directory = './model/benign/{}'.format(args.dataset)
    bn_model_files = os.listdir(benign_directory)
    bn_model_count = len(bn_model_files)
    print("Total benign models to load: {}".format(bn_model_count))

    for i in range(bn_model_count):
        print("Loading model {}/{}".format(i + 1, bn_model_count))  # 显示当前正在加载的模型编号和总数
        # Prepare model
        # benign_model_path =  os.path.join(os.getcwd(), "model", "benign", "model_{i}.pth")

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
        keep = np.asarray(keep)  # 将keep列表转换为NumPy数组
        print("First 10 elements of keep:", keep[:10])

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


        probs = []
        for idx in idxs:
            image = testset.__getitem__(idx)[0]

            with torch.no_grad():
                outputs_unsmoothed = benign_model(torch.unsqueeze(image.to(device), dim=0))
                _, predicted_unsmoothed = outputs_unsmoothed.max(1)

                image_smoothed = smooth(image.to(device), N=1024,sigma=args.sigma)
                outputs = benign_model(image_smoothed)
                _, predicted = outputs.max(1)

                prob = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))
                # print(f"Index: {idx}, Predicted Probabilities: {prob}")

                probs.append(prob)
        probs = np.asarray(probs)
        probs = np.mean(probs, axis=0)
        print(f"Mean Predicted Probabilities: {probs}")

        probs_max.append(np.amax(probs))
        print(f"Max Mean Predicted Probability: {np.amax(probs)}")

    probs_max = np.asarray(probs_max)
    print(probs_max)
    print("Array Size:", probs_max.size)  # 查看数组总元素个数

    stats_null_ranked = np.sort(probs_max)
    N = len(probs_max)
    m = int(N * args.beta)
    idx_thres = int(N - m - np.floor((N - m + 1) * args.theta))
    thres = stats_null_ranked[idx_thres]
    print(f"Threshold value: {thres}")

    filename = '{}_ownership_verification_sigma{}.txt'.format(args.dataset, args.sigma)
    f = open(filename, 'w')
    # 提前打开文件以记录所有模型的准确率
    print("model\tmax_delta\tmin_str", file=f, flush=True)
    # VSR and WCA
    actual = 0
    certified = 0
    total = 0

    watermarked_directory = './model/watermarked/{}_{}'.format(args.dataset, args.watermarked_type)
    wm_model_files = os.listdir(watermarked_directory)
    wm_model_count = len(wm_model_files)
    print("Total watermarked models to load: {}".format(wm_model_count))

    for i in range(wm_model_count):

        # Prepare model
        watermark_model_path = f'{watermarked_directory}/model_{i}.pth'
        watermark_model = load_model(args, watermark_model_path)
        watermark_model.eval()

        # Load trigger
        trigger = torch.load('./model/watermarked/{}_{}_trigger/trigger_{}'.format(args.dataset,args.watermarked_type, i))
        target_class = torch.load('./model/watermarked/{}_{}_trigger/target_class_{}'.format(args.dataset,args.watermarked_type, i))

        # 计算触发器的L2范数
        trigger_norm = torch.norm(trigger).item()
        print(f"*******触发器的L2范数是：{trigger_norm}")

        strs = []
        deltas = []
        prob_actual_aggre = 0
        for idx in idxs:
            image = testset.__getitem__(idx)[0]
            image_wm = torch.clamp(image + trigger, min=0, max=1)

            # Prob of image
            with torch.no_grad():
                image_smoothed = smooth(image.to(device), N=1024, sigma=args.sigma)
                outputs = watermark_model(image_smoothed)
                _, predicted = outputs.max(1)
                prob = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))

                prob_actual_aggre += prob[target_class]

                # Compute STR and delta
                image_wm_smoothed = smooth(image_wm.to(device), N=1024, sigma=args.sigma)
                outputs_wm = watermark_model(image_wm_smoothed.float())
                _, predicted_bd = outputs_wm.max(1)
                prob_bd = get_freq(predicted_bd.detach().cpu().numpy(), outputs_wm.size(1))
                deltas.append(torch.norm(image_wm - image).detach().cpu().numpy())
                strs.append(prob_bd[target_class])

        strs = np.asarray(strs)
        print(f"---所有计算的STR值: {strs}")
        deltas = np.asarray(deltas)
        print(f"---所有计算的delta值: {deltas}")
        Pi = np.min(strs)
        print(f"********最小STR的值是******：{Pi}")
        Delta = np.max(deltas)
        print(f"*******最大的delta值是******：{Delta}")

        if args.sigma * (Ginv(1 - thres) - Ginv(1 - Pi)) - Delta > 0:
            certified += 1
        prob_actual_aggre /= len(idxs)
        if prob_actual_aggre > thres:
            actual += 1
        total += 1
        print(f"{i}\t{Delta:.3f}\t{Pi:.3f}", file=f, flush=True)

    f.close()
    print("Completed all models.")
    print('WCA: %.3f' % (certified / total))
    print('VSR: %.3f' % (actual / total))




