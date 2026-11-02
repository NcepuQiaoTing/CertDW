import argparse
import os
import torch
import torch.nn as nn
from dataloader import get_cifar10, get_gtsrb
from modelloader import get_model

parser = argparse.ArgumentParser(description='Train benign models for evaluation')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, svhn, gtsrb')
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
if not os.path.isdir('./model/benign'):
    os.mkdir('./model/benign')
save_path = './model/benign/{}'.format(args.dataset)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

def main():

    if args.dataset == 'cifar10':
        trainset, trainloader, testset, testloader = get_cifar10(train_split_ration=0.8, testset_split_ration=0.5)
    elif args.dataset == 'gtsrb':
        trainset, trainloader, testset, testloader = get_gtsrb(train_split_ration=0.605, testset_split_ration=0.8)

    f = open('{}_benign_acc.txt'.format(args.dataset), 'w')
    print("model\tacc", file=f, flush=True)
    for i in range(args.n_model):
        model = get_model(args)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else torch.optim.SGD(model.parameters(), lr=args.lr)

        print("Training started")
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / (batch_idx + 1)}")
        # Evaluation
        print("Starting evaling")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print('Model: %s; acc: %.3f' % (i, acc))
        torch.save(model.state_dict(), f'{save_path}/model_{i}.pth')
        print(f"{i}\t{acc:.3f}", file=f, flush=True)
    f.close()
    print("Completed all models.")

if __name__ == '__main__':
    main()