import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,  # 输入图像尺寸
        patch_size: int = 16,  # 补丁尺寸
        in_chans: int = 3,  # 输入图像通道数
        embed_dim: int = 768,  # 补丁嵌入维度
        depth: int = 12,  # ViT 的深度
        num_heads: int = 12,  # 每个 ViT 块中的注意力头数
        mlp_ratio: float = 4.0,  # MLP 隐藏维度与嵌入维度的比例
        out_chans: int = 256,  # 输出通道数
        qkv_bias: bool = True,  # 如果为 True，则在查询、键、值中添加可学习偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 归一化层
        act_layer: Type[nn.Module] = nn.GELU,  # 激活函数
        use_abs_pos: bool = True,  # 如果为 True，则使用绝对位置嵌入
        use_rel_pos: bool = False,  # 如果为 True，则将相对位置嵌入添加到注意力图中
        rel_pos_zero_init: bool = True,  # 如果为 True，则初始化相对位置参数为零
        window_size: int = 0,  # 窗口注意力块的窗口大小
        global_attn_indexes: Tuple[int, ...] = (),  # 使用全局注意力的块的索引
    ) -> None:
        super().__init__()
        self.img_size = img_size  # 输入图像尺寸

        # 补丁嵌入层
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),  # 补丁大小
            stride=(patch_size, patch_size),  # 步幅
            in_chans=in_chans,  # 输入通道数
            embed_dim=embed_dim,  # 嵌入维度
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # 使用预训练图像大小初始化绝对位置嵌入
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        # 创建 ViT 块
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # 颈部连接层
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 补丁嵌入
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed  # 添加绝对位置嵌入

        # 通过 ViT 块
        for blk in self.blocks:
            x = blk(x)

        # 通过颈部连接层
        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """带有窗口注意力和残差传播块支持的Transformer块"""

    def __init__(
        self,
        dim: int,  # 输入通道数
        num_heads: int,  # 每个ViT块中的注意力头数
        mlp_ratio: float = 4.0,  # MLP隐藏维度与嵌入维度的比例
        qkv_bias: bool = True,  # 如果为True，则在查询、键、值中添加可学习偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 归一化层
        act_layer: Type[nn.Module] = nn.GELU,  # 激活函数
        use_rel_pos: bool = False,  # 如果为True，则将相对位置嵌入添加到注意力图中
        rel_pos_zero_init: bool = True,  # 如果为True，则初始化相对位置参数为零
        window_size: int = 0,  # 窗口注意力块的窗口大小。如果为0，则使用全局注意力
        input_size: Optional[Tuple[int, int]] = None,  # 用于计算相对位置参数大小的输入分辨率
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)  # 第一个归一化层
        self.attn = Attention(  # 注意力层
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)  # 第二个归一化层
        self.mlp = MLPBlock(  # 多层感知器块
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size  # 窗口大小

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # 残差连接
        x = self.norm1(x)  # 第一个归一化层

        # 窗口分割
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)  # 注意力层

        # 反向窗口分割
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x  # 残差连接
        x = x + self.mlp(self.norm2(x))  # 第二个归一化层和MLP块

        return x


class Attention(nn.Module):
    """带有相对位置编码的多头注意力块。"""

    def __init__(
        self,
        dim: int,  # 输入通道数
        num_heads: int = 8,  # 注意力头数
        qkv_bias: bool = True,  # 如果为True，则在查询、键、值中添加可学习偏置
        use_rel_pos: bool = False,  # 如果为True，则添加相对位置编码到注意力图中
        rel_pos_zero_init: bool = True,  # 如果为True，则初始化相对位置参数为零
        input_size: Optional[Tuple[int, int]] = None,  # 用于计算相对位置参数大小的输入分辨率
    ) -> None:
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = head_dim**-0.5  # 缩放因子

        # 查询、键、值的线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 投影层
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos  # 是否使用相对位置编码
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # 初始化相对位置嵌入
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape  # 获取输入张量的形状
        # 对输入进行查询、键、值的线性变换
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # 分解成查询、键、值
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # 计算注意力得分
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # 添加相对位置编码
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        # 注意力加权值与值相乘，并重塑输出张量
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        # 通过投影层
        x = self.proj(x)

        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    将输入分割成非重叠的窗口，并进行必要的填充。
    Args:
        x (tensor): 输入张量，形状为 [B, H, W, C]。
        window_size (int): 窗口大小。

    Returns:
        windows: 分割后的窗口，形状为 [B * num_windows, window_size, window_size, C]。
        (Hp, Wp): 分割前的填充高度和宽度
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    将窗口取消分割为原始序列并移除填充。
    Args:
        windows (tensor): 输入张量，形状为 [B * num_windows, window_size, window_size, C]。
        window_size (int): 窗口大小。
        pad_hw (Tuple): 填充的高度和宽度（Hp，Wp）。
        hw (Tuple): 原始的高度和宽度（H，W）在填充之前。

    Returns:
        x: 取消分割后的序列，形状为 [B, H, W, C]。
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    根据查询和键的相对位置获取相对位置嵌入。
    Args:
        q_size (int): 查询 q 的大小。
        k_size (int): 键 k 的大小。
        rel_pos (Tensor): 相对位置嵌入（L，C）。

    Returns:
        根据相对位置提取的位置嵌入。
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # 如果需要，进行相对位置插值。
    if rel_pos.shape[0] != max_rel_dist:
        # 进行相对位置插值。
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # 如果q和k的形状不同，则缩放坐标的短长度。
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args(参数):
        attn (Tensor): 注意力图。
        q (Tensor): 注意力层中的查询 q，形状为 (B, q_h * q_w, C)。
        rel_pos_h (Tensor): 高度轴的相对位置嵌入 (Lh, C)。
        rel_pos_w (Tensor): 宽度轴的相对位置嵌入 (Lw, C)。
        q_size (Tuple): 查询 q 的空间序列尺寸，形状为 (q_h, q_w)。
        k_size (Tuple): 键 k 的空间序列尺寸，形状为 (k_h, k_w)。

    Returns:
        attn (Tensor): 添加了相对位置嵌入的注意力图。
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    # 计算相对位置嵌入
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    # 计算相对位置编码的乘积
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # 将相对位置编码添加到注意力图中
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    图像到补丁嵌入。
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): 投影层的卷积核大小。
            stride (Tuple): 投影层的步幅。
            padding (Tuple): 投影层的填充大小。
            in_chans (int): 输入图像通道数。
            embed_dim (int): 补丁嵌入维度。
        """
        super().__init__()
        # 定义投影层
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过投影层
        x = self.proj(x)
        # 将通道维移到最后一个维度
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
