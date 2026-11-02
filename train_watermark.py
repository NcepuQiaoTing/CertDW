import argparse
import os
import torch
import torch.nn as nn
from dataloader import get_cifar10, get_gtsrb, get_watermark_cifar10, get_watermark_gtsrb
from modelloader import get_model

parser = argparse.ArgumentParser(description='Train benign models for evaluation')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, svhn, gtsrb')
parser.add_argument('--watermark_type', type=str, required=True, help='badnet, blended(patch), blended(noise)')
parser.add_argument('--target_class', type=int, default=1, help='Target class of watermark samples')
parser.add_argument('--optimizer', type=str, default='adam', help='sgd, adam')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--n_model', type=int, default=50, help='Number of benign models')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help="'cuda' if torch.cuda.is_available() else 'cpu'")

args = parser.parse_args()
device = args.device

if not os.path.isdir('./model'):
    os.mkdir('./model')
if not os.path.isdir('./model/watermark'):
    os.mkdir('./model/watermark')
save_path = './model/watermark/{}_{}'.format(args.dataset, args.watermark_type)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(f'{save_path}_trigger'):
    os.mkdir(f'{save_path}_trigger')

def main():

    if args.dataset == 'cifar10':
        trainset, _, testset, testloader = get_cifar10(testset_split_ration=0.5)
    elif args.dataset == 'gtsrb':
        trainset, trainloader, testset, testloader = get_gtsrb(testset_split_ration=0.5)


    f = open('{}_watermark_acc_wsr.txt'.format(args.dataset), 'w')
    print("model\tacc\twsr", file=f, flush=True)

    i = 0
    while i < args.n_model:
        if args.dataset == 'cifar10':
            watermark_trainloader, watermark_testloader, trigger = get_watermark_cifar10(args, trainset, testset)
        elif args.dataset == 'gtsrb':
            watermark_trainloader, watermark_testloader, trigger = get_watermark_gtsrb(args, trainset, testset)
        model = get_model(args)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else torch.optim.SGD(
            model.parameters(), lr=args.lr)

        print("Training started")
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(watermark_trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}: Loss = {running_loss / (batch_idx + 1):.3f}")
        # Evaluation
        print("Starting evaling")
        model.eval()
        correct = 0
        total = 0 #
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        # wm evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(watermark_testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        wsr = 100. * correct / total

        if acc > 60 and wsr > 90:
            torch.save(model.state_dict(), f'{save_path}/model_{i}.pth')
            torch.save(trigger, f'{save_path}_trigger/trigger_{i}')
            print('Model: %s; acc: %.3f; asr: %.3f' % (i, acc, wsr))
            i += 1
        else:
            print('Failed attack: acc: %.3f; asr: %.3f ' % (acc, wsr))
            # 写入模型的准确率到文件
        print(f"{i}\t{acc:.3f}\t{wsr:.3f}", file=f, flush=True)

    f.close()
    print("Completed all models.")

if __name__ == '__main__':
    main()



