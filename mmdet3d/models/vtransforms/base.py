from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]

# 生成BEV特征图的分辨率、第一个网格的中心坐标和网格数
# dx：BEV特征图在各坐标轴上的分辨率，即步长
# bx：BEV特征图在各坐标轴上第一个网格的中心坐标
# nx：BEV特征图在各坐标轴上的网格数
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        # dbound是一个三元组，这个三元组用于定义生成BEV图像的深度范围和分辨率
        # 具体来说，dbound = (dmin, dmax, dstep)，其中dmin表示最小深度，dmax表示最大深度，dstep表示深度步长，即分辨率。
        # 这里的深度指的是相机坐标系下的z轴坐标，即相机到物体的距离。
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # C是BEV特征图的通道数，即BEV特征图的最后一维的大小
        self.C = out_channels
        self.frustum = self.create_frustum()
        # D是BEV特征图的深度，即BEV特征图的第一维的大小
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    # 创建形如[D,fH,fW,3]的相机视椎体（四棱锥台）张量frustum，视椎体由D*fH*fW个点组成，即BEV特征图的几何结构
    # 张量的前三维D,fH,fW分别描述了三维视椎体的深度（层数）、高度和宽度，与BEV特征图的形状一致
    # 张量的最后一维描述了视椎体中每个点的三维坐标(v,u,z)，其中v和u表示对应于图像的像素坐标，z表示相机坐标系下点到坐标原点的距离
    # 【归纳】frustum张量描述了BEV空间与相机空间的对应关系，即BEV空间中的每个点对应于相机空间中的一个三维点：(D,fH,fW) -> (v,u,z)
    # 【归纳】(D,fH,fW)是BEV空间的坐标，(v,u,z)是相机空间，v和u的单位是像素
    # 【归纳】由于BEV空间是通过对图像均匀采样建立的，因此BEV空间此时本质上就是相机空间，只不过多了深度而已    
    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size   # iH和iW是相机图像的高和宽
        fH, fW = self.feature_size # fH和fW是BEV图像（输出特征图）的高和宽

        # 构造视椎体中每个点的z坐标，来自dbound，即BEV特征图的深度范围和分辨率信息，表示相机坐标系下点到坐标原点的距离，即深度。
        # ds是形状为[D,fH,fW]的张量，其中D是深度方向上的分辨率，即深度步长
        # 即ds由D个fH*fW的二维张量堆叠而成，每个二维张量内的元素值都相同：相机坐标系下点到坐标原点的距离，即深度。
        ds = (
            # arange(start, end, step)函数用于生成[start, end)区间内，步长为step的一维张量
            torch.arange(*self.dbound, dtype=torch.float) # 生成深度范围内的一维深度值 [D]
            .view(-1, 1, 1)                               # [D] -> [D,1,1]
            .expand(-1, fH, fW)                           # [D,1,1] -> [D,fH,fW]
        )
        D, _, _ = ds.shape

        # 构造视椎体中每个点的x坐标和y坐标，即（v,u），来自图像的像素坐标
        # xs和ys是形状为[D,fH,fW]的张量，其中D是深度方向上的分辨率，即深度步长
        # xs和ys由D个完全相同的fH*fW的二维张量堆叠而成，每个二维张量内的元素值分别表示图像宽度和高度方向上的像素坐标。
        # xs中的每一行元素都是从0到iW-1之间均匀分布的fW个数值，也就是说对图像像素宽度进行均匀采样，使得xs的形状为[D,fH,fW]
        xs = (
            # linspace(start, end, num)函数用于生成[start, end]区间内，num个元素的一维张量
            # 函数生成从 0 到 iW-1 之间均匀分布的 fW 个数值，也就是说对图像宽度进行均匀采样，使得xs的形状为[D,fH,fW]
            torch.linspace(0, iW - 1, fW, dtype=torch.float) # 生成图像宽度范围内的一维宽度值 [fW]
            .view(1, 1, fW)                                  # [fW] -> [1,1,fW]
            .expand(D, fH, fW)                               # [1,1,fW] -> [D,fH,fW]
        )
        # ys中的每一列元素都是从0到iH-1之间均匀分布的fH个数值，也就是说对图像像素高度进行均匀采样，使得ys的形状为[D,fH,fW]
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # 把ds、xs和ys沿着最后一个维度拼接起来，获得每个点的相机空间坐标，即(v,u,z)
        # 拼接后frustum的最终形状为[D,fH,fW,3]，其中3表示(v,u,z)
        # stack(tensors, dim)函数用于沿着指定的维度拼接张量
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    # 将相机视椎体frustum的像素坐标转换成雷达坐标系并返回，即geometry
    # 图像有多个，每个图像与雷达的转换矩阵不同，因此转换完成后的geometry扩充了两个维度B和N
    # 视椎体的形状从[D,fH,fW,3]扩展成[B,N,D,fH,fW,3]，其中B是batch size，N是图像个数
    # 【归纳】像素坐标转成雷达坐标之后，geometry本质上表征了相机空间与雷达空间的映射关系
    # 【归纳】图像有多个，雷达只有一个，因此geometry描述了图像与雷达的多对一映射关系
    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,  # 相机到雷达的旋转矩阵
        camera2lidar_trans, # 相机到雷达的平移向量
        intrins,            # 相机内参矩阵
        post_rots,          # 图像增强变换矩阵-旋转
        post_trans,         # 图像增强变换矩阵-平移
        **kwargs,           # 其他参数
    ):
        # camera2lidar_trans的形状是[B,N,1], B是batch size，N是图像个数，1是平移向量的维度
        B, N, _ = camera2lidar_trans.shape

        # self.frustum是表征相机视椎体的张量，形状是[D,fH,fW,3]
        # 其中第3维是像素坐标（v,u,z），其中z是相机坐标系下点到坐标原点的距离，即深度。

        # 使用图像增强变换矩阵对视椎体进行变换，提高模型的鲁棒性和泛化能力，该矩阵在训练时随机生成，推理时设为单位矩阵
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # 这段代码是将点的坐标从像素坐标系转到雷达坐标系。
        # points形状是[B,N,D,H,W,3]，转换前第5维是像素坐标（v,u,z），其中z是相机坐标系下点到坐标原点的距离，即深度。
        # 从像素坐标系转到相机坐标系，首先需要将每个点的v和u坐标分别乘以z坐标，z坐标不变。
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], # v和u坐标分别乘以z坐标  
                points[:, :, :, :, :, 2:3], # z坐标不变
            ),
            5, # 在第5维上拼接
        )
        # 然后再乘以相机内参矩阵的逆矩阵，从而得到点在相机坐标系中的坐标。
        # 转换后points形状不变，第5维变成了相机坐标（x,y,z）
        # 然后再乘以相机到雷达的旋转矩阵
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 最后再加上相机到雷达的平移矩阵，从而得到点在雷达坐标系中的坐标。
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # 如果有额外的旋转矩阵和平移矩阵，也需要将它们应用到点上。
        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        # 最后返回视椎体在雷达坐标系中的坐标，形状是[B,N,D,H,W,3]
        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    # 将geometry中每个点的雷达系坐标转成网格坐标，然后将geometry和特征图x同时展平
    # 展平之后，geometry和特征图x的B,N,D,fH,fW维度信息基本上都丢失了，只剩下每个点的网格坐标和特征信息
    # 【归纳】geometry和x的前五维的形状完全相同，也就说，每个点的网格坐标和特征图中的点是一一对应的
    # 【归纳】只要特征图x中的点和geometry中的网格坐标一一对应，geometry的目的就达到了，前五维的形状也就不重要了
    # 【归纳】BEV特征图中每个点的特征在x中，网格坐标在geometry中，两者一一对应，就可以将BEV特征图中的每个点的特征提取出来
    @force_fp32()
    def bev_pool(self, geom_feats, x):
        
        # geom_feats是BEV特征图中的每个点在雷达坐标系中的坐标（x,y,z）
        # x是从图像以及点云深度信息提取到的特征图

        # Nprime是BEV特征图中的总点数
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # 将提取到的深度和图像特征展平
        # flatten x
        x = x.reshape(Nprime, C)

        # geom_feats形状是[B,N,D,H,W,3]，其中第5维是视椎体中的每个点从像素坐标转成雷达坐标后的坐标（x,y,z）

        # flatten indices
        # 将视椎中的点在雷达坐标系中的坐标（x,y,z）转换到BEV特征图中的网格索引（x,y,z）
        # self.bx表示BEV特征图的左下角点的x坐标，self.dx表示BEV特征图的网格大小
        # 通过减去左下角点的坐标并除以网格大小，可以得到每个点对应的网格坐标。
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        # 将视椎体中的点展平
        geom_feats = geom_feats.view(Nprime, 3)
        # 创建批次索引：batch_ix
        # 这段代码的作用是创建一个形状为 (Nprime, 1) 的张量 batch_ix，其中 Nprime 是展平后的总点数，表示每个点属于哪个批次。
        # 这个张量是通过将B个形状为 (Nprime // B, 1) 的张量拼接起来得到的，每个小张量的元素值都是对应的批次编号，//表示整除。
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        # 将batch_ix拼接到展平的BEV特征图中，拼接后geom_feats的形状为 (Nprime, 4)，前三列分别是x,y,z，第4列是批次索引。
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # 把BEV特征图中的点的网格坐标限制在BEV特征图的范围内
        # self.nx表示BEV特征图在各个维度上的网格数目
        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # FIXME: 为什么这里要把nx[0]也就是X轴传递给H，nx[1]也就是Y轴传递给W？
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,                #图像数据，包括RGB、深度等信息  [B, M, C, imgH, imgW]
        points,             #点云数据                      [B][N, 3]
        sensor2ego,         #传感器到车辆坐标系的变换矩阵   
        lidar2ego,          #激光雷达到车辆坐标系的变换矩阵
        lidar2camera,       #激光雷达到相机坐标系的变换矩阵
        lidar2image,        #激光雷达到图像坐标系的变换矩阵，隐含了内参矩阵K  [M, 4, 4]
        cam_intrinsic,      #相机内参           
        camera2lidar,       #相机到激光雷达坐标系的变换矩阵
        img_aug_matrix,     #图像数据增强变换矩阵       [M, 4, 4]
        lidar_aug_matrix,   #点云数据增强变换矩阵       [4, 4]
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        # 获得点云的batch_size，也就是点云的个数
        # points是一个list，其中每个元素都是一个形状为[n, 3]的二维点云张量，n是点的个数，3是点的坐标
        batch_size = len(points)
        # 定义深度图，是一个5维张量，分别是[batch_size、图像数、通道数、img高、img宽]，注意和img一致，是为了便于后续提取特征
        # img是一个5维张量，分别是[batch_size、图像数、通道数、img高、img宽]，img.shape[1]是图像数
        # 其中的*是解包操作，将构成self.image_size元组的所有元素解包到torch.zeros函数的参数列表中
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )

        # 这段代码是将点云投影到图像上，并根据投影后的像素坐标更新深度图。
        # 具体来说，该代码段接收点云张量points，以及一些变换矩阵和参数。
        # 首先，对点云进行几何变换（如反向变换、点云到图像的投影等）将其映射到图像上；
        # 然后，获取投影后每个点的像素坐标，并根据这些坐标将其对应的深度值更新到深度图中，最终输出深度图depth。
        for b in range(batch_size): # 遍历list中的每个点云张量
            cur_coords = points[b][:, :3] # 取得第b个点云张量中所有行的前三列，即所有点的坐标，cur_coords的形状是[n, 3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # 将点云坐标从激光雷达坐标系转换到车辆坐标系，转换完成后cur_coords的坐标形状为[3, n]
            # 另一种说法是使用雷达增强矩阵对点云进行旋转、缩放、平移等操作，提高模型的鲁棒性和泛化能力，该矩阵在训练时随机生成，推理时设为单位矩阵
            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]  # 先平移
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(  
                cur_coords.transpose(1, 0)  # 后旋转
            )

            # 将点云坐标从车辆坐标系转换到相机坐标系
            # 雷达到相机的变换矩阵已经隐含了相机内参矩阵，所以后面不再需要再乘以相机内参矩阵
            # 由于有多个图像，因此雷达到相机的变换矩阵cur_lidar2image是一个3维张量，形状为[m, 4, 4]，m是图像的个数
            # 转换完成后cur_coords的形状为[m, 3, n]，每个图像对应一个[3, n]的坐标矩阵
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) # 先旋转
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)  # 后平移

            # 取得每个点的深度值，并将3D相机坐标转成2D像素坐标
            # get 2d coords
            dist = cur_coords[:, 2, :] # 取得每个图像坐标矩阵[3, n]中的第三行（z坐标），即每个点的深度值，形状为[m, n]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5) # 将深度值z限制在[1e-5, 1e5]之间
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # 每个点的x和y坐标除以深度z，将3D相机坐标转成2D图像坐标

            # 雷达到相机的变换矩阵已经隐含了相机内参矩阵，所以这里不再需要显式的乘以相机内参矩阵

            # 使用图像增强矩阵对图像进行旋转、缩放、平移等操作，提高模型的鲁棒性和泛化能力，该矩阵在训练时随机生成，推理时设为单位矩阵
            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords) # 先旋转
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)  # 后平移
            cur_coords = cur_coords[:, :2, :].transpose(1, 2) # 将坐标矩阵的形状从[m, 3, n]转成[m, n, 3]

            # 将每个图像对应的[n, 3]坐标矩阵中的x和y坐标对调，即转成H*W，并删除z坐标，输出形状为[m, n, 2]的坐标矩阵
            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            # 判断坐标矩阵中的每个点是否落在图像内
            # on_img是一个形状为[m, n]的布尔值张量，表示每个图像对应的[n, 2]像素坐标矩阵中的每个点是否落在图像内。
            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            # 如果落在图像内，则更新深度图
            # 在这一步骤中，原本乱序的点云深度值被重新排序，按照[B, N, C, H, W]的顺序排列，没有对应点云的位置深度值为0
            # depth是一个形状为(B, N, 1, H, W)的五维张量，表示每个像素的深度值，B表示batch大小，N表示图像个数，H表示图像高度，W表示图像宽度。
            # depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]]表示在第b个批次和第c个通道中，BEV图像上有效像素的深度值。
            for c in range(on_img.shape[0]):  # 对于每个图像c
                masked_coords = cur_coords[c, on_img[c]].long() # 取得落在图像内的像素坐标，且取整，形状为[n, 2]
                masked_dist = dist[c, on_img[c]] # 取得落在图像内的深度值，形状为[n]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist # 更新深度图

        # 将相机视椎体frustum的像素坐标转换成雷达坐标系并返回，建立图像坐标系和雷达坐标系之间的映射关系geometry
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # 从图像和深度图中提取特征，即特征图
        # 提取到的特征形状与geometry前五维的形状完全一致：[batch_size、图像数、深度、img高、img宽、通道数]
        # 至此，特征图与geometry的形状一致，元素一一对应，可以直接进行池化操作
        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)

        # 【归纳】在DepthLSSTransform中，将点云数据转换为BEV特征图后，使用BEVPool对BEV特征图及其网格坐标进行池化。
        # BEVPool通过将BEV特征图分割成多个体素块，并按照体素块的排列顺序对其进行排序，从而将点云和图像数据中的信息汇聚到相应的体素块中。
        # 通过这种方式，可以将原始点云和图像数据中的丰富信息转换为BEV特征图中的相应特征，从而使后续的神经网络模型能够更好地利用这些信息进行任务处理。

        return x
