import torchvision.models as models
import torch.nn as nn


def res_block_50(args):
    block = models.resnet50(args.pretrained)

    return nn.Sequential(
        block.conv1,
        block.bn1,
        block.relu,
        block.maxpool,
        block.layer1,
        block.layer2,
        block.layer3,
        block.layer4)
