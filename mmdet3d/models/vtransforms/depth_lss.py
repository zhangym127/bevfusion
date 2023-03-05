from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:  #表示该函数没有返回值
        #调用父类的构造函数
        super().__init__(  
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential( 
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    # 从图像和深度图中提取特征
    # x（即img）是一个5维张量，分别是[batch_size、图像数、通道数、img高、img宽]
    # d（即depth），是一个5维张量，分别是[batch_size、图像数、通道数、img高、img宽]
    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        # 把img和depth的前两维合并，即[batch_size、图像数]合并为一个维度
        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # 从深度图中提取特征
        # 通过多次卷积操作提取深度信息的特征，使得深度图能够更好地表达物体在三维空间中的形状和位置
        d = self.dtransform(d)
        # 将转换后的深度图和图像拼接起来，并通过depthnet网络提取特征。
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        # 取特征图的前D维，即深度信息，进行softmax归一化，转成深度概率图
        depth = x[:, : self.D].softmax(dim=1)
        # 用深度概率图和表示颜色的特征图后C维相乘，实现深度信息和颜色信息的融合
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        # 将特征图的维度还原回[batch_size、图像数、通道数、深度、img高、img宽]
        x = x.view(B, N, self.C, self.D, fH, fW)
        # 将特征图的维度进行转置，变成[batch_size、图像数、深度、img高、img宽、通道数]
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x