# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# 设置随机种子
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")
# 设置线程数
os.environ["OMP_NUM_THREADS"] = "4"  # 设置并行计算时使用的线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # 设置OpenBLAS库使用的线程数
os.environ["MKL_NUM_THREADS"] = "6"  # 设置MKL库使用的线程数
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # 设置VecLib库使用的线程数
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # 设置NumExpr库使用的线程数


# 定义函数show_mask用于显示掩码
def show_mask(mask, ax, random_color=False):
    if random_color:  # 如果指定了随机颜色
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # 生成随机颜色
    else:  # 否则使用默认颜色
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])  # 默认颜色
    h, w = mask.shape[-2:]  # 获取掩码的高度和宽度
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # 生成掩码图像
    ax.imshow(mask_image)  # 在指定的Axes对象上显示掩码图像


# 定义函数show_box用于显示边界框
def show_box(box, ax):
    x0, y0 = box[0], box[1]  # 获取边界框左上角坐标
    w, h = box[2] - box[0], box[3] - box[1]  # 获取边界框的宽度和高度
    ax.add_patch(  # 添加矩形框到Axes对象
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)  # 绘制矩形框
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root  # data_root=data/npy/CT_Abd
        self.gt_path = join(data_root, "gts")  # ground truth真实值所在路径：self.gt_path=data/npy/CT_Abd/gts
        self.img_path = join(data_root, "imgs")  # 训练图像所在路径：self.img_path=data/npy/CT_Abd/imgs
        #  通过 glob.glob 函数获取存放标签文件的路径下所有的 .npy 文件，并按照文件路径进行排序。
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        #  筛选出在图像路径下也存在相应文件名的标签文件，确保标签文件与图像文件对应。这一步重点在“筛选”，不一定所有标签都有对应需要训练的值
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        #  设置边界框偏移量，存疑
        self.bbox_shift = bbox_shift
        #  终端输出标签图片数量
        print(f"number of images: {len(self.gt_path_files)}")

    #  返回数据集中样本的数量。
    def __len__(self):
        return len(self.gt_path_files)

    #  获取指定索引处的样本数据
    def __getitem__(self, index):
        # 获取图像文件名，并加载对应的图像数据，返回的图像数据为 (1024, 1024, 3) 的 NumPy 数组 [0,1]
        img_name = os.path.basename(self.gt_path_files[index])  #  self.gt_path_files列表中索引为index的元素，代表一个文件的完整路径。os.path.basename(path)用于从完整的文件路径中提取文件名
        img_1024 = np.load(  # join函数发生了路径拼接
            join(self.img_path, img_name), "r", allow_pickle=True # pickle一个序列化模块参数，先不管
        )  # load是将图像文件转化为一个像素点矩阵，规格为(1024, 1024, 3)
        #  将图像数据的形状转换为 (3, H, W)，其中 H 和 W 分别表示图像的高度和宽度
        img_1024 = np.transpose(img_1024, (2, 0, 1))  #重新排列维度，第0个维度是第2个维度，第1个维度是第0个维度...因为许多模型的期望输入形状是(通道数，高度，宽度)
        assert (  #保证像素值在0到1，确保图像已完成归一化
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(  #发生了下采样，标签值为256*256，原来是1024*1024*3
            self.gt_path_files[index], "r", allow_pickle=True
        )  # 加载标签文件，返回的标签数据为多个类别的标签，形状为 (256, 256) 的 NumPy 数组 multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error"  # + self.gt_path_files[index] + self.npy_files[index] # 这里原来代码可能有点问题，先注释掉
        )
        label_ids = np.unique(gt)[1:]  # 排除背景标签，需要好好理解一下
        gt2D = np.uint8(  #  二值化处理，gt中是一个有好多类别的感兴趣区域，通过二值化处理后，gt2d变成一个只对某一个类别感兴趣的区域
            gt == random.choice(label_ids.tolist())  #  通过随机选择类别，可以确保模型不会对某一特定类别产生偏好，从而提高模型的泛化能力。
        )  # 从标签数据中随机选择一个类别作为目标类别，并将标签数据转换为二维数组，其中目标类别对应的像素值为 1，其他像素值为 0(256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        # 计算ROI区域大小
        y_indices, x_indices = np.where(gt2D > 0)
        xL, xR = np.min(x_indices), np.max(x_indices)
        yL, yR = np.min(y_indices), np.max(y_indices)
        alpha = xR - xL
        beta = yR - yL
        omegaROI = alpha * beta
        # 计算图像整体大小
        H, W = gt2D.shape
        omegaImg = H * W
        # 设置基于ROI面积的比例因子
        thetaOmega = omegaROI / omegaImg
        # 计算ROI区域的框嵌入边界信息比例关系：
        xi = alpha / beta
        # 设置扰动偏移量
        muOmega = 3.0  # 将模型对框嵌入提示信息的感受野变小的方向调整
        deltaAlpha = alpha * thetaOmega * muOmega
        deltaBeta = beta * thetaOmega * muOmega / xi
        # 限制扰动偏移量最大不超过20
        max_shift = 20
        deltaAlpha = min(deltaAlpha, max_shift)
        deltaBeta = min(deltaBeta, max_shift)

        # 应用自适应随机扰动
        xL_prime = np.random.randint(xL - int(deltaAlpha), xL + int(deltaAlpha) + 1)
        yL_prime = np.random.randint(yL - int(deltaBeta), yL + int(deltaBeta) + 1)
        xR_prime = np.random.randint(xR - int(deltaAlpha), xR + int(deltaAlpha) + 1)
        yR_prime = np.random.randint(yR - int(deltaBeta), yR + int(deltaBeta) + 1)

        # # 确保扰动后的坐标不会超出图像的边界
        # xL_prime = max(0, min(xL_prime, W - alpha))
        # yL_prime = max(0, min(yL_prime, H - beta))
        # xR_prime = max(xL_prime + alpha - 1, min(xR_prime, W - 1))
        # yR_prime = max(yL_prime + beta - 1, min(yR_prime, H - 1))

        # 将边界框的坐标保存到数组bboxes中
        bboxes = np.array([xL_prime, yL_prime, xR_prime, yR_prime])
        return (
            torch.tensor(img_1024).float(),  # 返回图像数据
            torch.tensor(gt2D[None, :, :]).long(),  # 返回标签数据
            torch.tensor(bboxes).float(),  # 返回边界框坐标
            img_name,  # 返回图像文件名
        )


# 进行数据集类的健全性测试
tr_dataset = NpyDataset("data/npy/CT_Abd")
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)  # 创建数据加载器，批量加载数据集。设置批量大小为 8，打乱数据顺序
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape, bboxes.shape)
    '''
        image 存储图像数据
        gt 存储标签数据
        bboxes 存储边界框数据
        names_temp 存储图像文件名
    '''
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))  # 创建一个具有两个子图的画布
    idx = random.randint(0, 7)  # 随机选择一个索引，用于在当前批次中选择样本
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")  # 在第一个子图中显示图像、标签掩码和边界框
    # 设置标题为图像文件名
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")  # 在第二个子图中显示图像、标签掩码和边界框
    # 设置标题为图像文件名
    axs[1].set_title(names_temp[idx])

    # plt.show()  待更改

    # 调整子图之间的间距，并保存画布为图片文件。随后关闭画布，以便下一次循环重新创建新的画布
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# 设置参数解析器（parser）
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",  # 指定训练数据的路径
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")  # 指定任务名称
parser.add_argument("-model_type", type=str, default="vit_b")  # 指定模型类型
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/MedSAM-ViT-B-20241117-1029/medsam_model_best.pth"  # 指定检查节点
)
parser.add_argument('-device', type=str, default='cuda:0')  # 如果没有可用的 GPU，则会使用 CPU
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"  # 指示是否加载预训练模型，默认为 True。"加载预训练模型"是指使用已经在其他数据集上训练过的模型作为新任务的起点。
)
parser.add_argument("-pretrain_model_path", type=str, default="./work_dir/MedSAM-ViT-B-20241117-1029/medsam_model_best.pth")  # 预训练模型路径
parser.add_argument("-work_dir", type=str, default="./work_dir")  # 指定工作目录
# 训练
parser.add_argument("-num_epochs", type=int, default=10)  # 训练轮数设置为10，原来是1000
parser.add_argument("-batch_size", type=int, default=1)  # 批量大小 batch_size
parser.add_argument("-num_workers", type=int, default=0)  # 工作线程 并行线程数量为0
# 添加优化器参数
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"  # 权重衰减 正则化的一种方法，利用L2范数限制模型复杂度，weight_decay的系数设置为0.01
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"  # 学习率
)
parser.add_argument(  # 监视训练参数 不使用wandb监视模型训练，wandb是什么存疑
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp")  # 启用AMP（自动混合精度），AMP是一种在深度学习训练中用于提高性能和减少内存使用的技术，它结合了16位和32位浮点数的计算，以优化训练速度和效率。
parser.add_argument(  # 不从检查点恢复训练
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")  #  使用第一个GPU训练
args = parser.parse_args()  # 解析命令行参数，并将结果保存到 args 变量中

if args.use_wandb:  #  如果使用 wandb则进行下面步骤，这里先不看
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")  # 标识本次训练的运行日期
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder  冻结提示编码器的参数，使其在训练过程中不会更新参数
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):  # 定义前向传播方法 forward()，接受图像和边界框作为输入
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder  使用无梯度更新的上下文，通过提示编码器对边界框进行编码，得到稀疏和密集的提示特征表示
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        # 使用图像特征表示和提示特征表示，通过掩码解码器生成低分辨率的掩码预测
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # 通过插值操作将低分辨率的掩码预测插值到原始分辨率
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():  # __file__ 是一个特殊的内置变量，它包含了当前脚本文件的路径。这个变量只在脚本文件内部有效，通常用于获取当前文件的路径。
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    # 实例化SAM 模型，根据 SAM 模型的组件构建MedSAM 模型，将其移动到指定的设备上，并设置为训练模式
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472  计算并打印模型中的总参数数量，结果为93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252  计算并打印模型中的可训练参数数量，结果为93729252

    # 优化器
    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(  # 采用AdamW优化器
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252  图像编码器和掩码解码器中可训练参数的数量
    # 分割损失函数
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # 交叉熵损失函数
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs  # 训练轮数
    iter_num = 0  # 迭代次数 初始值为0
    losses = []  # 损失列表
    best_loss = 1e10  # 最佳损失值 best_loss变量用于记录训练过程中的最低损失值，初始值设置为一个非常大的数（1e10）
    train_dataset = NpyDataset(args.tr_npy_path)  # 训练数据集

    print("Number of training samples: ", len(train_dataset))  # 输出训练集样本数量
    train_dataloader = DataLoader(  # 数据加载器
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0  # 起始轮次为第零轮开始训练
    if args.resume is not None:  # resume为空这个代码块不运行，先不看
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:  # 初始化自动混合精度（AMP）的梯度缩放器
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):  # 训练循环
        epoch_loss = 0  # 初始化轮次损失
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()  #  在每个批次开始时，清除优化器的梯度信息。
            boxes_np = boxes.detach().cpu().numpy()  #  将边界框数据 boxes 转换为 NumPy 数组
            image, gt2D = image.to(device), gt2D.to(device)  #  将图像和标签数据移动到指定的设备
            if args.use_amp:  # 未启用，先不看
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:  #  直接进行前向传播，计算损失，进行反向传播，更新优化器，然后清零梯度。
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()  # 累积损失
            iter_num += 1  # 更新迭代次数

        epoch_loss /= step  #  计算本轮次的平均损失
        losses.append(epoch_loss)  # 添加到损失列表中
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = medsam_model.state_dict()

            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()  # 在python3.12中工作
