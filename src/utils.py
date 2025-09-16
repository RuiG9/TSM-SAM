import os
import numpy as np
from osgeo import gdal
import rasterio
from affine import Affine
from rasterio.transform import from_origin

def read_img(filename):
    """gdal读取路径内的图像文件并提取相关信息。
    
    Args:
        filename (str): 图像文件的路径。
        
    Returns:
        tuple: 包含图像数据数组、高度、宽度、波段数、仿射变换参数、投影信息。
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        print(f"Failed to open file: {filename}")
        return None
        
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(float)
    im_band = im_data.shape[0] if len(im_data.shape) > 2 else 1
    
    dataset = None
    return im_data, im_height, im_width, im_band, im_geotrans, im_proj

def combine_features(spectral_data, spectral_bands=None):
    """合并光谱数据，确保波段维度在最后。
    
    Args:
        spectral_data (ndarray): 光谱数据的数组。
        spectral_bands (list): 选中的光谱波段。
        
    Returns:
        ndarray: 组合后的特征数组，波段在最后一个维度。
    """
    def select_bands(data, bands):
        band_dim = np.argmin(data.shape)
        
        if bands is None:
            bands = list(range(data.shape[band_dim]))
            
        if band_dim == 0:
            selected = data[bands, :, :]
        elif band_dim == 1:
            selected = data[:, bands, :]
        else:
            selected = data[:, :, bands]
            
        if band_dim != 2:
            selected = np.moveaxis(selected, 0, -1)
        return selected

    selected_spectral = select_bands(spectral_data, spectral_bands)
    return selected_spectral

def calculate_image_change_Local(image_before, image_after, method='difference'):
    """计算两幅影像的变化。
    
    Args:
        image_before (ndarray): 灾害前的影像。
        image_after (ndarray): 灾害后的影像。
        method (str): 计算方法，包括'difference'、'ratio'和'rate_of_change'。
        
    Returns:
        ndarray: 影像变化结果。
    """
    if method == 'difference':
        return image_after - image_before
    elif method == 'ratio':
        return image_after / (image_before + 1e-10)
    elif method == 'rate_of_change':
        difference = image_after - image_before
        return difference / (np.abs(image_before) + 1e-10)
    else:
        raise ValueError("Unsupported method. Choose 'difference', 'ratio', or 'rate_of_change'.")

def prepare_features(input_band, roi_mask, roiID=1):
    """处理并准备用于模型输入的特征数据。
    
    Args:
        input_band (ndarray): 输入波段数据。
        roi_mask (array): 作物感兴趣区域的掩膜。
        roiID (int): ROI标识。
        
    Returns:
        tuple: 处理后的全数据和roi区域数据。
    """
    roi_mask_feature = roi_mask.flatten()
    roi_ind = np.where(roi_mask_feature == roiID)
    other_ind = np.where(roi_mask_feature == 0)

    height, width, num_bands = input_band.shape
    num_pixels = height * width
    
    input_data = np.zeros((num_pixels, num_bands), dtype=float)
    input_roi = np.zeros((len(roi_ind[0]), num_bands), dtype=float)

    for i in range(num_bands):
        band_data = input_band[:, :, i].flatten()
        band_data[np.isnan(band_data)] = 0
        input_data[:, i] = band_data
        input_roi[:, i] = band_data[roi_ind]

    return input_data, input_roi, roi_ind, other_ind

def save_gee_image_as_tiff(image_data, geotransform, projection, scale, filename):
    """将图像数据保存为TIFF文件。
    
    Args:
        image_data (ndarray): 图像的NumPy数组数据。
        geotransform (list): 仿射变换参数。
        projection (str): CRS投影信息。
        scale (float): 像素比例。
        filename (str): 输出文件路径。
    """
    new_geotransform = [
        geotransform[0],
        scale,
        geotransform[2],
        geotransform[3],
        geotransform[4],
        -scale
    ]

    if image_data.ndim == 2:
        num_bands = 1
        height, width = image_data.shape
        image_data = image_data[:, :, np.newaxis]
    elif image_data.ndim == 3:
        height, width, num_bands = image_data.shape
    else:
        raise ValueError("Image data must be either 2D or 3D array")

    transform = Affine(new_geotransform[1], new_geotransform[2], new_geotransform[0],
                      new_geotransform[4], new_geotransform[5], new_geotransform[3])

    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=image_data.dtype,
        crs=projection,
        transform=transform
    ) as dst:
        for i in range(num_bands):
            dst.write(image_data[:, :, i], i + 1) 