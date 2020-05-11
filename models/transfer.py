from torchvision import models
from torch import nn
from torchaudio import transforms


class Resnet50Music(nn.Module):
    def __init__(self):
        super(Resnet50Music, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.speclayer = transforms.MelSpectrogram()
        self.initial_layer = nn.Sequential(
             nn.Conv2d(1, 128, 3, 2),
             nn.BatchNorm2d(128)
            )

        self.resnet._modules['conv1'] = nn.Conv2d(128, 64, 3, 1)
        self.resnet._modules['bn1'] = nn.BatchNorm2d(64)
        in_ftrs= self.resnet._modules['fc'].in_features
        self.resnet._modules['fc'] = nn.Linear(in_ftrs, 4)

    def forward(self, x):
        x = x.unsqueeze(1)
        in_spec = self.speclayer(x)
        out = self.initial_layer(in_spec)
        out = self.resnet(out)
        return out