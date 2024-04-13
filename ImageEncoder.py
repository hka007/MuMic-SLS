import timm
import torch
from torch import nn
from config import CFG
from torchvision.models import vit_b_16


class ImageEncoder(nn.Module):
    def __init__(
            self, model_name=CFG.image_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable


    def forward(self, x):
        features = self.model(x)

        return features


class ImageEncoder_2(nn.Module):
    def __init__(
            self, model_name=CFG.image_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = vit_b_16(
            pretrained
        )

        self.model.heads = torch.nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        features = self.model(x)
        return features
