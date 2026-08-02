import os
import urllib.request
import zipfile
from pathlib import Path

# Fix relative import when running directly
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.config import MODELS_DIR

YOLO_URL = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face-lindevs.onnx"
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    
    # Progress indicator
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({read_so_far}/{total_size} bytes)")
        else:
            sys.stdout.write(f"\rDownloaded {read_so_far} bytes")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, dest_path, reporthook=report)
    print("\nDownload finished.")

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 1. Download YOLO Face model
    yolo_dest = MODELS_DIR / "yolov8n-face.onnx"
    if not yolo_dest.exists():
        try:
            download_file(YOLO_URL, yolo_dest)
        except Exception as e:
            print(f"Failed to download YOLO Face model: {e}")
            print("Trying alternative download from latest tag...")
            try:
                alt_url = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx"
                download_file(alt_url, yolo_dest)
            except Exception as e2:
                print(f"Alternative download failed too: {e2}")
    else:
        print("YOLO Face model already exists.")

    # 2. Download and extract ArcFace model (from buffalo_s.zip)
    arcface_dest = MODELS_DIR / "arcface.onnx"
    if not arcface_dest.exists():
        zip_dest = MODELS_DIR / "buffalo_s.zip"
        if not zip_dest.exists():
            try:
                download_file(BUFFALO_S_URL, zip_dest)
            except Exception as e:
                print(f"Failed to download buffalo_s: {e}")
                return
        
        print("Extracting buffalo_s.zip...")
        try:
            with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                # In buffalo_s.zip, the ArcFace model might be named w600k_mobi.onnx or w600k_r50.onnx
                # Let's inspect the files in the zip
                namelist = zip_ref.namelist()
                print("Files in zip:", namelist)
                
                # We look for a file ending with w600k_mobi.onnx or w600k_r50.onnx or just .onnx with face recognition
                recognition_model = None
                for name in namelist:
                    if "w600k" in name and name.endswith(".onnx"):
                        recognition_model = name
                        break
                
                if recognition_model:
                    print(f"Found recognition model: {recognition_model}")
                    # Extract it
                    zip_ref.extract(recognition_model, MODELS_DIR)
                    # Move/Rename it to arcface.onnx
                    extracted_path = MODELS_DIR / recognition_model
                    extracted_path.rename(arcface_dest)
                    print(f"Successfully extracted and renamed to {arcface_dest}")
                else:
                    print("Could not find a w600k ONNX model in the zip.")
        except Exception as e:
            print(f"Failed to extract zip file: {e}")
        finally:
            if zip_dest.exists():
                zip_dest.unlink()  # Remove zip to clean up space
    else:
        print("ArcFace model already exists.")

if __name__ == "__main__":
    main()
