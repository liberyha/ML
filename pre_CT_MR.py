# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d  # 用于3D连通区域分析

# convert nii image to npz files, including original image and corresponding masks
# 定义文件路径和相关参数
modality = "CT"  # 图像类型（CT或其他）
anatomy = "Abd"  # 解剖部位和数据集名称的组合
img_name_suffix = "_0000.nii.gz"  # 图像文件名后缀
gt_name_suffix = ".nii.gz"  # ground truth文件名后缀
prefix = modality + "_" + anatomy + "_"  # 生成文件前缀

nii_path = "data/FLARE22Train/images"  # path to the nii images(原始图像路径)
gt_path = "data/FLARE22Train/labels"  # path to the ground truth(原始图像路径)
npy_path = "data/npy/" + prefix[:-1]  # 转换后的NumPy数据保存路径
# 创建保存路径
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)

image_size = 1024  # 图像尺寸
voxel_num_thre2d = 100  # 2D阈值
voxel_num_thre3d = 1000  # 3D阈值
# 读取所有文件名并进行筛选
names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")

# set label ids that are excluded
# 排除特定标签
remove_label_ids = [
    12  # 排除的标签ID，比如12代表十二指肠
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
# 设置窗口水平和宽度
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = 40  # 仅用于CT图像
WINDOW_WIDTH = 400  # 仅用于CT图像

# %% save preprocessed images and masks as npz files
# 循环处理每个文件
for name in tqdm(names[:40]):  # use the remaining 10 cases for validation(只处理前40个文件)
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # 删除指定的标签
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    # 读取肿瘤标签并转换为实例分割
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # exclude the objects with less than 1000 pixels in 3D
    # 移除体积小于1000像素的对象
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # remove small objects with less than 100 pixels in 2D slices
    # 在2D切片中移除像素数量小于100的对象
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # 找到非零切片
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        # 图像预处理
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]

        # 保存为npz文件
        np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # save the image and ground truth as nii files for sanity check;
        # 保存为nii文件进行检查
        # they can be removed
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        )
        # # 将每个CT图像保存为npy文件
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            resize_img_skimg = transform.resize(
                img_3c,
                (image_size, image_size),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
            resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            gt_i = gt_roi[i, :, :]
            resize_gt_skimg = transform.resize(
                gt_i,
                (image_size, image_size),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
            resize_gt_skimg = np.uint8(resize_gt_skimg)
            assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
            np.save(
                join(
                    npy_path,
                    "imgs",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_img_skimg_01,
            )
            np.save(
                join(
                    npy_path,
                    "gts",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_gt_skimg,
            )
