import sys
from network import ShadowModel, SimpleNet
import torch
def get_model(args):
    if args.dataset == 'cifar10':
        model = ShadowModel()
    elif args.dataset == 'gtsrb':
        model = SimpleNet()
    else:
        sys.exit('Incorrect dataset name!')
    model = model.to(args.device)
    return model

def load_model(args, model_path):
    model = get_model(args)
    print(model)
    model.load_state_dict(torch.load(model_path))
    return model


