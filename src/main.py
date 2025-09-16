import os
import numpy as np
from utils import (
    read_img, combine_features, calculate_image_change_Local,
    prepare_features, save_gee_image_as_tiff
)
from threshold_segmentation import threshold_analysis
from sam_segmentation import sam_analysis, merge_segmentations, load_sam_model
from sklearn.ensemble import IsolationForest
from osgeo import gdal
from segment_anything_hq import sam_model_registry, SamPredictor

def train_isolation_forest(input_data, test_size=0.2, random_state=42):
    """
    Train the Isolation Forest model and make predictions.
    
    Args:
        input_data (ndarray): Input data.
        test_size (float): Test set ratio.
        random_state (int): Random seed.
        
    Returns:
        ndarray: Prediction results.
    """
    model = IsolationForest(
        n_estimators=50,
        max_samples='auto',
        contamination=float(0.1),
        max_features=1.0,
        random_state=random_state
    )
    model.fit(input_data)
    return model.decision_function(input_data)

def read_and_process_flood_data(params):
    """
    Read and process flood data.
    
    Args:
        params (dict): Parameter dictionary.
        
    Returns:
        tuple: (ROI_pre_decision_result, ROIvalues, metadata)
    """
    if not (params.get('filename1') and params.get('filename2')):
        raise ValueError("Missing required filenames")

    try:
        before_values, row, column, band, geotrans, proj = read_img(params['filename1'])
        after_values, *_ = read_img(params['filename2'])
        
        values = calculate_image_change_Local(before_values, after_values, method=params.get('change_method', 'rate_of_change'))
        input_band = combine_features(values, params.get('spectral_bands'))

        if params['roi_type'] == 1:
            ROIvalues = np.ones((row, column))
        elif params['roi_type'] == 2 and params.get('filename3'):
            ROIvalues1, *_ = read_img(params['filename3'])
            ROIvalues = ROIvalues1.astype(np.int32).squeeze()
        else:
            raise ValueError("Invalid ROI type or missing ROI file")

        input_data, input_roimasked, roi_ind, other_ind = prepare_features(input_band, ROIvalues, 1)
        
        pred_decision = train_isolation_forest(input_roimasked)
        ROI_pre_decision = np.zeros((row * column), dtype=float)
        ROI_pre_decision[roi_ind] = pred_decision
        ROI_pre_decision_result = ROI_pre_decision.reshape(row, column)

        metadata = {
            'row': row,
            'column': column,
            'band': band,
            'geotrans': geotrans,
            'proj': proj
        }

        return ROI_pre_decision_result, ROIvalues, metadata

    except Exception as e:
        print(f"Error processing images: {e}")
        raise

def flood_analysis_pipeline(params):
    """
    Main flood analysis pipeline.
    
    Args:
        params (dict): Parameter dictionary.
        
    Returns:
        tuple: (Merge_mask, output_paths)
    """
    # Ensure output directory exists
    os.makedirs(params['output_folder_path'], exist_ok=True)
    
    # Validate SAM model parameters
    if not params.get('SAM_checkpoint') or not params.get('model_type'):
        raise ValueError("Please specify 'SAM_checkpoint' and 'model_type' in params, and ensure the weight file is downloaded")
    
    # Read and process data
    ROI_pre_decision_result, ROIvalues, metadata = read_and_process_flood_data(params)
    print("[Progress] Data reading and preprocessing completed.")

    # Threshold analysis
    threshold_results = threshold_analysis(ROI_pre_decision_result, ROIvalues, params)
    Threshold_merge = threshold_results["Threshold_merge"]
    print("[Progress] Threshold segmentation completed.")

    # SAM analysis
    SAM_mask = sam_analysis(ROI_pre_decision_result, Threshold_merge, ROIvalues, params)
    print("[Progress] SAM segmentation analysis completed.")

    # Merge segmentation results
    Merge_mask = merge_segmentations(Threshold_merge, SAM_mask)
    print("[Progress] Segmentation merging completed.")

    # Export results
    output_paths = {}
    for label, mask in [('Merge', Merge_mask), ('SAM', SAM_mask), ('TSM', Threshold_merge)]:
        output_file = f"{params['output_folder_path']}/{params['regionname']}_{label}_mask_output.tif"
        
        if label == 'SAM' or label == 'TSM':
            mask = (mask > 0).astype(np.uint8)

        save_gee_image_as_tiff(
            mask,
            metadata['geotrans'],
            metadata['proj'],
            0.00026949,  # scale parameter
            output_file
        )
        output_paths[label] = output_file
    print("[Progress] Result export completed.")

    return Merge_mask, output_paths

if __name__ == "__main__":
    print("[Progress] Flood analysis pipeline started.")
    # 示例参数配置
    params = {
        'regionname': 'Example',
        'change_method': 'rate_of_change',
        'roi_type': 1,
        'spectral_bands': [0, 1, 3],
        'filename1': 'flood_detection\data\BeforeFlooding.tif', 
        'filename2': 'flood_detection\data\AfterFlooding.tif',   
        'filename3': None,
        'combined_threshold': 13,
        'SAM_checkpoint': 'flood_detection\pretrained_checkpoint\sam_hq_vit_tiny.pth',
        'model_type': 'vit_tiny',
        'num_iterations': 20,
        'total_samples': 40,
        'min_samples_per_class': 10,
        'mask_threshold': 10,
        'mode': 2,
        'visualize': True,
        'output_folder_path': 'output' 
    }

    try:
        flood_mask, output_paths = flood_analysis_pipeline(params)
        print("[Progress] Flood analysis pipeline completed successfully!")
        print("Output files:")
        for label, path in output_paths.items():
            print(f"{label}: {path}")
    except Exception as e:
        import traceback
        print("[Error] Exception occurred during analysis:")
        traceback.print_exc() 