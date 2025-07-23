import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):  # 创建一个多层感知器（MLP）块
    def __init__(
        self,
        embedding_dim: int,  # 嵌入维度，即输入的特征维度。
        mlp_dim: int,  # MLP 层的维度
        act: Type[nn.Module] = nn.GELU,  # 激活函数
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)  # 线性层，将输入特征映射到 mlp_dim 维度
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)  # 线性层，将 mlp_dim 维度的输出映射回 embedding_dim 维度
        self.act = act()  # 激活函数，对 self.lin1 的输出应用激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义数据在这个块中的传递方式
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa

class LayerNorm2d(nn.Module):  # 创建一个二维的 Layer Normalization（层归一化）
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:  # num_channels：通道数，输入的特征通道数eps：归一化中的一个小常数，默认为1e-6
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义数据在这个归一化层中的传递方式
        u = x.mean(1, keepdim=True)  # 输入数据在第一维度（通道维度）上的均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 输入数据减去均值后的平方，再在第一维度上取平均得到的方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 将输入数据减去均值并除以标准差进行归一化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
