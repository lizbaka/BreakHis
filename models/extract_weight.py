import torch
import networks

ckpts = ['./ckpt/resnet50-bin.pth', './ckpt/densenet201-sub.pth']
models = [networks.ResNet50(num_classes=2), networks.DenseNet201(num_classes=8)]

for ckpt, model in zip(ckpts, models):
    model.load_state_dict(torch.load(ckpt)['model_state_dict'])
    torch.save({'model_state_dict': model.state_dict()}, ckpt.removesuffix('.pth') + '-reduced.pth')
