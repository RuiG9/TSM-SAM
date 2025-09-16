import os
import requests
from tqdm import tqdm
import hashlib
import argparse

MODEL_URLS = {
    'vit_h': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth',
    'vit_l': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth',
    'vit_b': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth'
}

MODEL_CHECKSUMS = {
    'vit_h': 'a7bf3b02f3ebf1c7b0701e1d7fb2e431',
    'vit_l': '85d08c41baf10c4a02d9cc5095b3fbbb',
    'vit_b': '3155e4c3e700d7bb48e3de9b8c290a67'
}

def download_file(url, filepath, chunk_size=8192):
    """
    从指定URL下载文件并显示进度条。
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as f, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def get_file_md5(filepath):
    """
    计算文件的MD5值。
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def verify_model_file(filepath, expected_md5):
    """
    验证模型文件的完整性。
    """
    if not os.path.exists(filepath):
        return False
    actual_md5 = get_file_md5(filepath)
    return actual_md5 == expected_md5

def download_sam_model(model_type='vit_l', save_dir='models'):
    """
    下载并验证SAM模型权重文件。
    """
    if model_type not in MODEL_URLS:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(MODEL_URLS.keys())}")
    os.makedirs(save_dir, exist_ok=True)
    filename = f'sam_hq_{model_type}.pth'
    filepath = os.path.join(save_dir, filename)
    if verify_model_file(filepath, MODEL_CHECKSUMS[model_type]):
        print(f"Model file already exists and verified: {filepath}")
        return filepath
    print(f"Downloading {model_type} model...")
    download_file(MODEL_URLS[model_type], filepath)
    if verify_model_file(filepath, MODEL_CHECKSUMS[model_type]):
        print(f"Model downloaded and verified successfully: {filepath}")
        return filepath
    else:
        os.remove(filepath)
        raise RuntimeError("Downloaded model file verification failed. Please try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HQ-SAM model weights.")
    parser.add_argument('--model_type', type=str, required=True, choices=['vit_l', 'vit_b', 'vit_h'], help='Model type to download (vit_l, vit_b, vit_h)')
    parser.add_argument('--save_dir', type=str, default='pretrained_checkpoint', help='Directory to save the model weights')
    args = parser.parse_args()
    download_sam_model(model_type=args.model_type, save_dir=args.save_dir) 