import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        
        self.resnet = models.resnet50()
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.num_classes = num_classes
        
        self.resnet = models.resnet101()
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.num_classes = num_classes
        
        self.densenet = models.densenet121()
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return x
    

class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        self.num_classes = num_classes
        
        self.densenet = models.densenet201()
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return x
    

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.num_classes = num_classes
        
        self.vgg11 = models.vgg11()
        
        num_features = self.vgg11.classifier[6].in_features
        self.vgg11.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.vgg11(x)
        return x


class VGG19_bn(nn.Module):
    def __init__(self, num_classes):
        super(VGG19_bn, self).__init__()
        self.num_classes = num_classes
        
        self.vgg19_bn = models.vgg19_bn()
        
        num_features = self.vgg19_bn.classifier[6].in_features
        self.vgg19_bn.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.vgg19_bn(x)
        return x


class ResNeXt_101_32x8d(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXt_101_32x8d, self).__init__()
        self.num_classes = num_classes
        
        self.resnext = models.resnext101_32x8d()
        
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnext(x)
        return x
    

network_dict = {
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'DenseNet121': DenseNet121,
    'DenseNet201': DenseNet201,
    'VGG11': VGG11,
    'VGG19_bn': VGG19_bn,
    'ResNeXt_101_32x8d': ResNeXt_101_32x8d
}
