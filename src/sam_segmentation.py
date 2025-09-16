import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label
from segment_anything_hq import sam_model_registry, SamPredictor
import torch


def show_mask(mask, ax, random_color=False):
    """显示分割掩膜。
    
    Args:
        mask (ndarray): 分割掩膜。
        ax: matplotlib轴对象。
        random_color (bool): 是否使用随机颜色。
    """
    color = np.random.random(3) if random_color else np.array([1, 0, 0])
    color = np.append(color, 0.6)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """显示点标注。
    
    Args:
        coords (ndarray): 点坐标。
        labels (ndarray): 点标签。
        ax: matplotlib轴对象。
        marker_size (int): 标记大小。
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 1], pos_points[:, 0], color='green', marker='.', 
              s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 1], neg_points[:, 0], color='red', marker='.', 
              s=marker_size, edgecolor='white', linewidth=1.25)

def extract_largest_regions(binary_mask, num_regions):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    sorted_indices = np.argsort(-stats[:, cv2.CC_STAT_AREA])
    combined_mask = np.zeros(binary_mask.shape, dtype=np.int64)
    # 只处理实际存在的前 num_regions 个区域（排除背景，背景是 sorted_indices[0]）
    max_regions = min(num_regions, num_labels - 1)
    for i in range(1, max_regions + 1):
        combined_mask[labels == sorted_indices[i]] = 1
    return combined_mask

def generate_points_from_mask(image, mask, roi_mask, total_samples=40, min_samples_per_class=10, mode=1):
    """根据掩膜生成采样点。
    
    Args:
        image (ndarray): 输入图像。
        mask (ndarray): 输入掩膜。
        roi_mask (ndarray): ROI掩膜。
        total_samples (int): 总采样点数。
        min_samples_per_class (int): 每类最小采样点数。
        mode (int): 采样模式。
        
    Returns:
        tuple: (points, labels)
    """
    valid_mask = roi_mask.squeeze() != 0
    positive_indices = np.argwhere((mask == 1) & valid_mask)
    negative_indices = np.argwhere((mask == 0) & valid_mask)
    
    num_positive_total = len(positive_indices)
    num_negative_total = len(negative_indices)
    total_indices = num_positive_total + num_negative_total

    if total_indices == 0:
        return [], []

    weight_positive = total_indices / (2 * num_positive_total) if num_positive_total > 0 else 0
    weight_negative = total_indices / (2 * num_negative_total) if num_negative_total > 0 else 0

    if mode == 1:
        num_negative = int(total_samples * weight_positive / (weight_positive + weight_negative))
        num_positive = total_samples - num_negative
    else:
        num_positive = int(total_samples * weight_positive / (weight_positive + weight_negative))
        num_negative = total_samples - num_positive

    num_positive = max(min_samples_per_class, min(num_positive, total_samples - min_samples_per_class))
    num_negative = max(min_samples_per_class, min(num_negative, total_samples - num_positive))

    chosen_positive = positive_indices[np.random.choice(len(positive_indices), num_positive, replace=False)] if len(positive_indices) >= num_positive else positive_indices
    chosen_negative = negative_indices[np.random.choice(len(negative_indices), num_negative, replace=False)] if len(negative_indices) >= num_negative else negative_indices

    points = np.vstack((chosen_positive, chosen_negative))
    labels = np.array([1] * len(chosen_positive) + [0] * len(chosen_negative))

    return points, labels

def merge_segmentations(seg_a, seg_b):
    """合并两个分割结果。
    
    Args:
        seg_a (ndarray): 第一个分割结果。
        seg_b (ndarray): 第二个分割结果。
        
    Returns:
        ndarray: 合并后的分割结果。
    """
    seg_a = seg_a > 0
    seg_b = seg_b > 0

    labeled_a = label(seg_a, connectivity=1)
    labeled_b = label(seg_b, connectivity=1)

    output_image = np.zeros_like(seg_a, dtype=np.uint8)
    max_label_a = labeled_a.max() + 1
    
    for i in range(1, max_label_a):
        mask_a = (labeled_a == i)
        overlap = mask_a & seg_b
        if np.any(overlap):
            output_image[mask_a] = 1

    return output_image

def load_sam_model(model_type, checkpoint_path):
    """
    加载HQ-SAM模型
    """
    return sam_model_registry[model_type](checkpoint=checkpoint_path)

def sam_analysis(ROI_pre_decision_result, Threshold_merge, ROIvalues, parameters):
    """使用SAM模型进行图像分割分析。
    
    Args:
        ROI_pre_decision_result (ndarray): 预处理的决策结果。
        Threshold_merge (ndarray): 阈值分割结果。
        ROIvalues (ndarray): ROI掩膜值。
        parameters (dict): 参数字典。
        
    Returns:
        ndarray: SAM分割结果。
    """
    binary_mask = np.where(Threshold_merge > 0, 255, 0).astype(np.uint8)
    largest_regions_mask = extract_largest_regions(binary_mask, num_regions=10)

    sam_checkpoint = parameters['SAM_checkpoint']
    model_type = "vit_l"
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(model)
    predictor.set_image(np.stack([ROI_pre_decision_result] * 3, axis=-1))

    accumulated_masks = None
    for _ in range(parameters['num_iterations']):
        points, labels = generate_points_from_mask(
            ROI_pre_decision_result, 
            largest_regions_mask, 
            ROIvalues, 
            parameters['total_samples'], 
            parameters['min_samples_per_class'], 
            parameters['mode']
        )
        
        masks, scores, logits = predictor.predict(
            point_coords=points, 
            point_labels=labels, 
            multimask_output=True
        )
        
        if accumulated_masks is None:
            accumulated_masks = np.zeros_like(masks[0], dtype=int)
        accumulated_masks += masks[0]

    SAM_mask = np.where(accumulated_masks >= parameters['mask_threshold'], 1, 0).astype(float)
    SAM_mask = SAM_mask * ROIvalues.astype(float)

    if parameters.get('visualize', True):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.stack([ROI_pre_decision_result] * 3, axis=-1))
        show_mask(SAM_mask, plt.gca())
        show_points(points, labels, plt.gca())
        plt.title('SAM Flooding')
        plt.axis('off')
        plt.show()

    return SAM_mask 