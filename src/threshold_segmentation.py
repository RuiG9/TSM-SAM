import numpy as np
import cv2
from skimage import filters, measure, morphology
from skimage.filters import threshold_isodata, threshold_li, threshold_mean, threshold_otsu, threshold_triangle, threshold_minimum, threshold_yen, threshold_local, threshold_niblack, threshold_sauvola
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

def concavity_threshold(image):
    thresh = np.percentile(image, 50)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def huang_threshold(image):
    thresh = threshold_yen(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def intermeans_threshold(image):
    # Intermeans (iterative selection threshold)
    thresh = image.mean()
    for _ in range(10):
        g1 = image[image <= thresh]
        g2 = image[image > thresh]
        if len(g1) == 0 or len(g2) == 0:
            break
        new_thresh = 0.5 * (g1.mean() + g2.mean())
        if np.abs(new_thresh - thresh) < 1e-3:
            break
        thresh = new_thresh
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def intermodes_threshold(image):
    # Intermodes: find two peaks and set threshold at valley
    hist, bin_edges = np.histogram(image.ravel(), bins=256)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist)
    if len(peaks) >= 2:
        valley = np.argmin(hist[peaks[0]:peaks[1]]) + peaks[0]
        thresh = bin_edges[valley]
    else:
        thresh = np.median(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def isodata_threshold(image, max_iter):
    if image.ndim != 2:
        raise ValueError("Image should be a 2D array")
    thresh = threshold_isodata(image, max_iter)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def Kittler_threshold(image):
    # Placeholder implementation, more complex algorithms can be used in practice
    thresh = np.percentile(image, 50)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def Li_threshold(image):
    thresh = threshold_li(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def max_entropy_threshold(image):
    # skimage.filters.threshold_yen 近似最大熵
    thresh = threshold_yen(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def mean_threshold(image):
    thresh = threshold_mean(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def median_threshold(image):
    thresh = np.median(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def otsu_threshold(image):
    thresh = threshold_otsu(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def percentile_threshold(image, percentile=20):
    thresh = np.percentile(image, percentile)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def minimum_threshold(image):
    thresh = threshold_minimum(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def triangle_threshold(image):
    thresh = threshold_triangle(image)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def tsai_threshold(image):
    # Placeholder implementation, more complex algorithms can be used in practice
    thresh = np.percentile(image, 50)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def zhenzhou_threshold(image):
    # Placeholder implementation, more complex algorithms can be used in practice
    thresh = np.percentile(image, 50)
    binary_image = np.where(image <= thresh, 1, 0)
    return thresh, binary_image

def threshold_analysis(ROI_pre_decision_result, ROIvalues, parameters):
    """
    Perform multiple threshold segmentation analyses on the preprocessed results.
    
    Args:
        ROI_pre_decision_result (ndarray): Preprocessed decision results.
        ROIvalues (ndarray): ROI mask values.
        parameters (dict): Parameter dictionary.
        
    Returns:
        dict: Dictionary containing thresholds and segmentation results.
    """
    thresholds = {}
    images = {}
    visualize = parameters.get('visualize')
    combined_threshold = parameters.get('combined_threshold')
    max_iter = 5

    # Apply all thresholding methods
    thresholds['thresh_concavity'], images['concavityROI'] = concavity_threshold(ROI_pre_decision_result)
    thresholds['thresh_huang'], images['huangROI'] = huang_threshold(ROI_pre_decision_result)
    thresholds['thresh_intermeans'], images['intermeansROI'] = intermeans_threshold(ROI_pre_decision_result)
    thresholds['thresh_intermodes'], images['intermodesROI'] = intermodes_threshold(ROI_pre_decision_result)
    thresholds['thresh_isodata'], images['isodataROI'] = isodata_threshold(ROI_pre_decision_result, max_iter)
    thresholds['thresh_Kittler'], images['KittlerROI'] = Kittler_threshold(ROI_pre_decision_result)
    thresholds['thresh_Li'], images['LiROI'] = Li_threshold(ROI_pre_decision_result)
    thresholds['thresh_max_entropy'], images['max_entropyROI'] = max_entropy_threshold(ROI_pre_decision_result)
    thresholds['thresh_mean'], images['meanROI'] = mean_threshold(ROI_pre_decision_result)
    thresholds['thresh_median'], images['medianROI'] = median_threshold(ROI_pre_decision_result)
    thresholds['thresh_otsu'], images['otsuROI'] = otsu_threshold(ROI_pre_decision_result)
    thresholds['thresh_percentile'], images['percentileROI'] = percentile_threshold(ROI_pre_decision_result, 20)
    thresholds['thresh_minimum'], images['minimumROI'] = minimum_threshold(ROI_pre_decision_result)
    thresholds['thresh_triangle'], images['triangleROI'] = triangle_threshold(ROI_pre_decision_result)
    thresholds['thresh_tsai'], images['tsaiROI'] = tsai_threshold(ROI_pre_decision_result)
    thresholds['thresh_zhenzhou'], images['zhenzhouROI'] = zhenzhou_threshold(ROI_pre_decision_result)

    # Merge threshold results
    matrix_3d_int = np.stack([img for img in images.values()], axis=-1)
    possibility_all = np.sum(matrix_3d_int, axis=-1)
    Threshold_merge = np.where(possibility_all >= int(combined_threshold), 1, 0)

    # Apply ROI mask
    for key in images:
        images[key] = images[key].astype(float) * ROIvalues.astype(float)

    if visualize:
        # Visualization code
        title = [key.replace('ROI', '') for key in images.keys()]
        colors = [(242/255, 235/255, 229/255), (6/255, 59/255, 101/255)]
        cmap = LinearSegmentedColormap.from_list("custom_binary", colors, N=2)

        n_images = len(images)
        ncols = 4
        nrows = (n_images + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten() if n_images > 1 else [axes]
        for i, (key, img) in enumerate(images.items()):
            ax = axes[i]
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(title[i], fontsize=10)
            ax.grid(False)
            ax.axis('off')
        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        # Create a legend
        legend_labels = ['Non-flooding', 'Flooding']
        patches = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(colors))]
        fig.legend(handles=patches, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
        # Set subplot spacing to 10% of subplot size
        fig.subplots_adjust(wspace=0.4, hspace=0.3)  # 0.4/4=10%, 0.3/3=10%
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

        plt.figure(figsize=(24, 12))
        plt.subplot(1, 2, 1)
        Threshold_score = plt.imshow(possibility_all, cmap='viridis')
        plt.title('Flooding Possibility')
        plt.grid(False)
        plt.axis('off')
        cbar = plt.colorbar(Threshold_score, fraction=0.028, pad=0.03)
        cbar.set_label('Scale')

        plt.subplot(1, 2, 2)
        plt.imshow(Threshold_merge, cmap=cmap)
        plt.title('Flooding')
        plt.grid(False)
        plt.axis('off')
        plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper right', ncol=1, facecolor='white', framealpha=0.5)
        plt.show()

    return {
        "thresholds": thresholds,
        "Threshold_merge": Threshold_merge
    } 