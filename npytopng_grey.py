import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def npy_to_png(npy_file_path, output_dir):
    # 加载 .npy 文件
    data = np.load(npy_file_path)

    # 从输入路径中提取文件名（不包括扩展名）
    base_filename = os.path.splitext(os.path.basename(npy_file_path))[0]

    # 构造输出文件的完整路径
    png_file_path = os.path.join(output_dir, base_filename + '.png')

    # 使用 matplotlib 将数据保存为 PNG 图像
    plt.imsave(png_file_path, data, cmap='gray')  # 使用灰度颜色图
    print(f"Saved {png_file_path}")

# 示例使用
npy_dir_path = 'data/npy/CT_Abd/imgs'  # .npy 文件的目录路径
output_dir = 'a_npytopng/img'  # 输出目录的路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取目录中的所有 .npy 文件
npy_files = glob.glob(os.path.join(npy_dir_path, '*.npy'))

# 遍历所有 .npy 文件并转换为 .png
for npy_file_path in npy_files:
    npy_to_png(npy_file_path, output_dir)