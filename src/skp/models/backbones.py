import re
import timm
import torch

from functools import partial

from .vmz.backbones import *


def check_name(name, s):
    return bool(re.search(s, name))


def create_backbone(name, pretrained, features_only=False, **kwargs):
    try:
        model = timm.create_model(name, pretrained=pretrained, 
                                  features_only=features_only, 
                                  num_classes=0, global_pool="")
    except Exception as e:
        assert name in BACKBONES, f"{name} is not a valid backbone"
        model = BACKBONES[name](pretrained=pretrained, features_only=features_only, **kwargs)
    with torch.no_grad():
        if check_name(name, r"x3d|csn|r2plus1d"):
            dim_feats = model(torch.randn((2,3,64,64,64))).size(1)
        else:
            dim_feats = model(torch.randn((2,3,128,128))).size(1)
    return model, dim_feats


def create_x3d(name, pretrained, features_only=False, z_strides=[1, 1, 1, 1, 1], **kwargs):
    if not pretrained:
        from pytorchvideo.models.x3d import create_x3d as _create_x3d
        model = _create_x3d(input_clip_length=16, input_crop_size=312, depth_factor=5)
    else:
        model = torch.hub.load("facebookresearch/pytorchvideo", model=name, pretrained=pretrained)   
    for idx, z in enumerate(z_strides):
        assert z in [1, 2], "Only z-strides of 1 or 2 are supported"
        if z == 2:
            if idx == 0:
                stem_layer = model.blocks[0].conv.conv_t
                w = stem_layer.weight
                w = w.repeat(1, 1, 3, 1, 1)
                in_channels, out_channels = stem_layer.in_channels, stem_layer.out_channels
                model.blocks[0].conv.conv_t = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
            else:                
                model.blocks[idx].res_blocks[0].branch1_conv.stride = (2, 2, 2)
                model.blocks[idx].res_blocks[0].branch2.conv_b.stride = (2, 2, 2)

    if features_only:
        model.blocks[-1] = nn.Identity()
        model = X3D_Features(model)
    else:
        model.blocks[-1] = nn.Sequential(
                model.blocks[-1].pool.pre_conv,
                model.blocks[-1].pool.pre_norm,
                model.blocks[-1].pool.pre_act,
            )

    return model


class X3D_Features(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.out_channels = [24, 24, 48, 96, 192]

    def forward(self, x):
        features = []
        for idx in range(len(self.model.blocks) - 1):
            x = self.model.blocks[idx](x)
            features.append(x)
        return features


BACKBONES = {
    "x3d_xs": partial(create_x3d, name="x3d_xs"),
    "x3d_s": partial(create_x3d, name="x3d_s"),
    "x3d_m": partial(create_x3d, name="x3d_m"),
    "x3d_l": partial(create_x3d, name="x3d_l"),
    "ir_csn_50": ir_csn_50,
    "ir_csn_101": ir_csn_101,
    "ir_csn_152": ir_csn_152,
    "ip_csn_50": ip_csn_50,
    "ip_csn_101": ip_csn_101,
    "ip_csn_152": ip_csn_152,
    "r2plus1d_34": r2plus1d_34
}