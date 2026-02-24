import urllib.request
import zipfile
import os

url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
zip_path = "liar_dataset.zip"
extract_path = "liar_dataset"

print(f"Downloading {url}...")
try:
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
    
    print(f"Extracting to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
    
    # List files
    print("\nExtracted files:")
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            print(os.path.join(root, file))
            
except Exception as e:
    print(f"Error: {e}")
