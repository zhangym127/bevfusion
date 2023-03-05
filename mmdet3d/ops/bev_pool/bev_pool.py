import torch

from . import bev_pool_ext

__all__ = ["bev_pool"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


# B: 批次数
# D: BEV特征网格的Z轴方向的网格数
# H: BEV特征网格的X轴方向的网格数
# W: BEV特征网格的Y轴方向的网格数
def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    # 此时特征图feats和对应的BEV网格坐标coords都已经展平，一一对应
    # 按照B、D、W、H的顺序，对特征图feats及其网格坐标coords升序排序
    # coords中记录
    ranks = (
        coords[:, 0] * (W * D * B)  #X轴方向的网格索引
        + coords[:, 1] * (D * B)    #Y轴方向的网格索引
        + coords[:, 2] * B          #Z轴方向的网格索引
        + coords[:, 3]              #批次索引
    )
    # argsort()返回的是从小到大的索引
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    # 使用CUDA完成池化操作
    # 通过coords中的网格索引，将feats中的特征值累加到对应的BEV网格中
    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
