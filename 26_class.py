import os
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segmentBooster import refineMask
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from skimage.measure import regionprops

# 定义多个数据目录
BASE_DATA_DIRS = [
    "/media/sdc2/lhx/SegmentBooster/data/mistake/2024.12.17problem",
    # "/media/sdc2/lhx/SegmentBooster/data/1213_yellow_grass/job_784",
    # "/media/sdc2/lhx/SegmentBooster/data/1213_yellow_grass/job_785",
    # "/media/sdc2/lhx/SegmentBooster/data/1213_yellow_grass/job_786",
    # "/media/sdc2/lhx/SegmentBooster/data/rgb",
]

SAM_CHECKPOINT = "/media/sdc2/lhx/SegmentBooster/sam_vit_h_4b8939.pth"  # SAM 检查点路径

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sam(sam_checkpoint):
    """加载SAM模型并初始化掩码生成器，使用第二份代码的参数"""
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=40,                # 修改为40
        pred_iou_thresh=0.82,              # 修改为0.82
        stability_score_thresh=0.90,       # 修改为0.90
        min_mask_region_area=15,           # 新增最小掩码区域面积
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,   # 修改为1
        output_mode="binary_mask"
    )
    return mask_generator

def merge_small_regions(segbooster_mask, original_mask, min_size=300):
    """使用第二份代码的合并小区域逻辑"""
    unique_values = np.unique(segbooster_mask)
    new_mask = segbooster_mask.copy()
    original_values = np.unique(original_mask[original_mask != 0])

    for val in unique_values:
        if val == 0:
            continue

        binary_mask = (segbooster_mask == val)
        labels, _ = ndimage.label(binary_mask)
        regions = regionprops(labels)

        for region in regions:
            if region.area < min_size:
                small_region_mask = (labels == region.label)
                overlap_region = original_mask[small_region_mask]
                unique_overlap = np.unique(overlap_region[overlap_region != 0])

                if len(unique_overlap) > 0 and region.area < 1000:
                    continue

                y1, x1, y2, x2 = region.bbox
                pad = 5
                y_min = max(0, y1 - pad)
                x_min = max(0, x1 - pad)
                y_max = min(segbooster_mask.shape[0], y2 + pad)
                x_max = min(segbooster_mask.shape[1], x2 + pad)

                neighborhood = segbooster_mask[y_min:y_max, x_min:x_max]
                unique_neighbors = np.unique(neighborhood)
                valid_neighbors = unique_neighbors[(unique_neighbors != val) & (unique_neighbors != 0)]

                if len(valid_neighbors) > 0:
                    neighbor_counts = [(n, np.sum(neighborhood == n)) for n in valid_neighbors]
                    most_common = max(neighbor_counts, key=lambda x: x[1])[0]
                    new_mask[small_region_mask] = most_common

    return new_mask

def fill_background(mask):
    zero_mask = (mask == 0)
    if not np.any(zero_mask):
        return mask

    filled_mask = np.copy(mask)

    while np.any(zero_mask):
        struct = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.grey_dilation(filled_mask, footprint=struct)
        filled_mask[zero_mask] = dilated[zero_mask]
        new_zero_mask = (filled_mask == 0)
        if np.array_equal(new_zero_mask, zero_mask):
            break
        zero_mask = new_zero_mask

    return filled_mask

def process_single_image(image_filename, directories, mask_generator):
    """处理单张图像并保存结果"""
    IMAGES_DIR, MASKS_DIR, REFINED_MASK_DIR, COMPARISON_DIR = directories
    image_name = os.path.splitext(image_filename)[0]
    image_path = os.path.join(IMAGES_DIR, image_filename)
    mask_path = os.path.join(MASKS_DIR, f"{image_name}.png")

    if not os.path.exists(mask_path):
        print(f"掩码文件不存在: {mask_path}，跳过此图像。")
        return

    try:
        image = np.array(Image.open(image_path).convert('RGB'))
        input_mask = np.array(Image.open(mask_path).convert('L'))

        sam_masks = sorted(mask_generator.generate(image), key=lambda x: x['area'], reverse=True)

        sam_mask = np.zeros_like(input_mask, dtype=np.int32)
        for mask_idx, mask in enumerate(sam_masks):
            if mask_idx >= 25:  # 修改为25
                break
            sam_mask[mask['segmentation']] = mask_idx + 1

        sam_mask = fill_background(sam_mask)

        refined_mask = refineMask(image_path, input_mask, mask_generator)
        refined_mask = fill_background(refined_mask)

        cleaned_mask = merge_small_regions(refined_mask, input_mask, min_size=300)  # 修改min_size为300
        cleaned_mask = merge_small_regions(cleaned_mask, input_mask, min_size=300)  # 仍然可以多次调用以进一步优化

        plt.figure(figsize=(20, 5))

        plt.subplot(141)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(sam_mask, cmap='tab20')
        plt.title('SAM MASK')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(input_mask, cmap='tab20')
        plt.title('Original Mask')
        plt.axis('off')

        plt.subplot(144)
        plt.imshow(cleaned_mask, cmap='tab20')
        plt.title('SegmentBooster Refined Mask')
        plt.axis('off')

        comparison_path = os.path.join(COMPARISON_DIR, f"{image_name}.png")
        plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.close()

        if cleaned_mask.max() <= 1:
            refined_mask_save = (cleaned_mask * 255).astype(np.uint8)
        else:
            refined_mask_save = cleaned_mask.astype(np.uint8)

        refined_mask_image = Image.fromarray(refined_mask_save)
        refined_mask_image.save(os.path.join(REFINED_MASK_DIR, f"{image_name}.png"))

        print(f"图像 {image_filename} 处理完成。比较图保存到: {comparison_path}，精炼后的掩码保存到: {os.path.join(REFINED_MASK_DIR, f'{image_name}.png')}")

    except Exception as e:
        print(f"处理图像 {image_filename} 时出错: {e}")

def process_data_directory(data_dir, sam_checkpoint):
    """处理单个数据目录中的所有图像"""
    IMAGES_DIR = os.path.join(data_dir, "JPEGImages")
    MASKS_DIR = os.path.join(data_dir, "mask")
    REFINED_MASK_DIR = os.path.join(data_dir, "refined_mask")
    COMPARISON_DIR = os.path.join(data_dir, "visualization")
    os.makedirs(REFINED_MASK_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    mask_generator = load_sam(sam_checkpoint)

    image_files = [f for f in os.listdir(IMAGES_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    if not image_files:
        print(f"目录 {IMAGES_DIR} 中未找到图像文件")
        return

    total_images = len(image_files)
    print(f"\n目录 {data_dir} 中找到 {total_images} 张图像文件。开始处理...")

    directories = (IMAGES_DIR, MASKS_DIR, REFINED_MASK_DIR, COMPARISON_DIR)

    start_time_total = time.time()

    use_parallel = False  # 是否使用并行处理

    if use_parallel:
        num_processes = min(cpu_count(), 4)  # 根据实际情况调整进程数
        with Pool(processes=num_processes) as pool:
            func = partial(process_single_image, directories=directories, mask_generator=mask_generator)
            pool.map(func, image_files)
    else:
        # 顺序处理
        for idx, image_filename in enumerate(image_files, 1):
            image_start_time = time.time()
            print(f"\n处理图像 {idx}/{total_images}: {image_filename}")
            process_single_image(image_filename, directories, mask_generator)
            image_end_time = time.time()
            image_elapsed = image_end_time - image_start_time
            print(f"图像 {image_filename} 处理耗时: {image_elapsed:.2f} 秒")

    # 记录总处理时间
    end_time_total = time.time()
    total_elapsed = end_time_total - start_time_total
    print(f"目录 {data_dir} 的所有图像处理完成。总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")

def main():
    """主程序入口，处理所有指定的数据目录"""
    overall_start_time = time.time()

    for data_dir in BASE_DATA_DIRS:
        if not os.path.isdir(data_dir):
            print(f"数据目录不存在或不是目录: {data_dir}，跳过。")
            continue
        process_data_directory(data_dir, SAM_CHECKPOINT)

    overall_end_time = time.time()
    overall_elapsed = overall_end_time - overall_start_time
    print(f"\n所有数据目录的图像处理完成。总耗时: {overall_elapsed:.2f} 秒 ({overall_elapsed/60:.2f} 分钟)")

if __name__ == "__main__":
    main()
