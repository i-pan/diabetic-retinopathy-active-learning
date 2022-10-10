import math
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbones
from . import segmentation
from .pooling import create_pool2d_layer, create_pool3d_layer
from .sequence import Transformer
from .tools import change_num_input_channels


class Conv3DReduce(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv_reduce = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=(kernel_size[0], 1, 1),
                                     padding=(0, kernel_size[1] // 2, kernel_size[2] // 2), bias=False)
        self.conv_reduce.weight = nn.Parameter(torch.ones_like(self.conv_reduce.weight) / np.prod(kernel_size))

    def forward(self, x):
        return self.conv_reduce(x.unsqueeze(1)).squeeze(2)


class Net2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 pool,
                 in_channels=3,
                 feature_reduction=None,
                 multisample_dropout=False,
                 load_pretrained_backbone=None,
                 freeze_backbone=False,
                 add_conv3d_reduce=None,
                 backbone_params={},
                 pool_layer_params={}):

        super().__init__()
        self.backbone, dim_feats = backbones.create_backbone(name=backbone, pretrained=pretrained, **backbone_params)
        self.pool_layer = create_pool2d_layer(name=pool, **pool_layer_params)
        if pool == "catavgmax": 
            dim_feats *= 2
        self.msdo = multisample_dropout
        if in_channels != 3:
            self.backbone = change_num_input_channels(self.backbone, in_channels)
        self.dropout = nn.Dropout(p=dropout)
        if isinstance(feature_reduction, int):
            # Use 1D grouped convolution to reduce # of parameters
            groups = math.gcd(dim_feats, feature_reduction)
            self.feature_reduction = nn.Conv1d(dim_feats, feature_reduction, groups=groups, kernel_size=1, stride=1, bias=False)
            dim_feats = feature_reduction
        self.classifier = nn.Linear(dim_feats, num_classes) 

        if load_pretrained_backbone: 
            # Assumes that model has a `backbone` attribute
            # Note: if you want to load the entire pretrained model, this is done via the
            # builder.build_model function
            print(f"Loading pretrained backbone from {load_pretrained_backbone} ...")
            weights = torch.load(load_pretrained_backbone, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get backbone only
            weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items() if 'backbone' in k}
            self.backbone.load_state_dict(weights)

        if freeze_backbone:
            print("Freezing backbone ...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        if add_conv3d_reduce:
            self.backbone = nn.Sequential(Conv3DReduce(in_channels=in_channels,
                                                       kernel_size=tuple(add_conv3d_reduce["kernel_size"])),
                                          self.backbone)

    def extract_features(self, x):
        features = self.backbone(x)
        features = self.pool_layer(features)
        if hasattr(self, "feature_reduction"):
            features = self.feature_reduction(features.unsqueeze(-1)).squeeze(-1)
        return features

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        if self.msdo:
            x = torch.mean(torch.stack([self.classifier(self.dropout(features)) for _ in range(5)]), dim=0)
        else:
            x = self.classifier(self.dropout(features))
        # Important nuance:
        # For binary classification, the model returns a tensor of shape (N,)
        # Otherwise, (N,C)
        x = x[:, 0] if self.classifier.out_features == 1 else x
        if return_features:
            return x, features
        return x

    def get_embedding_dim(self):
        return self.classifier.in_features


class Net3D(Net2D):

    def __init__(self, *args, **kwargs):
        z_strides = kwargs.pop("z_strides", [1,1,1,1,1])
        super().__init__(*args, **kwargs)
        self.pool_layer = create_pool3d_layer(name=kwargs["pool"], **kwargs.pop("pool_layer_params", {}))


class NetSegment2D(nn.Module):
    """ For now, this class essentially servers as a wrapper for the 
    segmentation model which is mostly defined in the segmentation submodule, 
    similar to the original segmentation_models.pytorch.

    It may be worth refactoring it in the future, such that you define this as
    a general class, then select your choice of encoder and decoder. The encoder
    is pretty much the same across all the segmentation models currently 
    implemented (DeepLabV3+, FPN, Unet).
    """
    def __init__(self,
                 architecture,
                 encoder_name,
                 encoder_params,
                 decoder_params,
                 num_classes,
                 dropout,
                 in_channels,
                 load_pretrained_encoder=None,
                 freeze_encoder=False,
                 deep_supervision=False,
                 pool_layer_params={},
                 aux_head_params={}):

        super().__init__()

        self.segmentation_model = getattr(segmentation, architecture)(
                encoder_name=encoder_name,
                encoder_params=encoder_params,
                dropout=dropout,
                classes=num_classes,
                deep_supervision=deep_supervision,
                in_channels=in_channels,
                **decoder_params
            )


        if load_pretrained_encoder: 
            # Assumes that model has a `encoder` attribute
            # Note: if you want to load the entire pretrained model, this is done via the
            # builder.build_model function
            print(f"Loading pretrained encoder from {load_pretrained_encoder} ...")
            weights = torch.load(load_pretrained_encoder, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.segmentation_model', '', k) : v for k,v in weights.items()}
            # Get encoder only
            weights = {re.sub(r'^encoder.', '', k) : v for k,v in weights.items() if 'backbone' in k}
            self.segmentation_model.encoder.load_state_dict(weights)

        if freeze_encoder:
            print("Freezing encoder ...")
            for param in self.segmentation_model.encoder.parameters():
                param.requires_grad = False


    def forward(self, x):
        return self.segmentation_model(x)


class NetSegment3D(NetSegment2D): 

    pass