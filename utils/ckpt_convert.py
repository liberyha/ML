# -*- coding: utf-8 -*-
import torch

# %% convert medsam model checkpoint to sam checkpoint format for convenient inference
sam_ckpt_path = "work_dir/SAM/sam_vit_b_01ec64.pth"  # SAM模型的检查点路径
medsam_ckpt_path = "work_dir/MedSAM/medsam_vit_b.pth"  # MedSam模型的检查点路径
save_path = "work_dir/SAM"  # 转换后的SAM模型检查点保存路径
multi_gpu_ckpt = True  # 如果模型是使用多GPU训练的，设置为True

sam_ckpt = torch.load(sam_ckpt_path)  # 加载SAM模型的检查点
medsam_ckpt = torch.load(medsam_ckpt_path)  # 加载MedSam模型的检查点
sam_keys = sam_ckpt.keys()  # 获取SAM模型检查点的键值列表
for key in sam_keys:
    if not multi_gpu_ckpt:
        sam_ckpt[key] = medsam_ckpt["model"][key]  # 如果不是多GPU模型，直接复制MedSam模型的相应键值到SAM模型
    else:
        sam_ckpt[key] = medsam_ckpt["model"]["module." + key]  # 如果是多GPU模型，使用"module."前缀并复制MedSam模型的相应键值到SAM模型

torch.save(sam_ckpt, save_path)  # 将转换后的SAM模型检查点保存到指定路径
