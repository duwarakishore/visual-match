import os
from bing_image_downloader import downloader
from PIL import Image
categories = [
    "wrist watch",
    "hand bag",
    "running shoes",
    "sunglasses",
    "laptop computer",
    "digital camera"
]
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
print(f"Dataset folder created: {DATASET_DIR}")

for item in categories:
    print(f"Downloading images for: {item}")
    downloader.download(
        item,
        limit=10,
        output_dir=DATASET_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("Download completed!")

print("Cleaning and preprocessing images")
combined_dir = os.path.join(DATASET_DIR, "all")
os.makedirs(combined_dir, exist_ok=True)

count = 0
for category in categories:
    folder_name = category.replace(" ", "_")
    category_path = os.path.join(DATASET_DIR, folder_name)
    
    if not os.path.exists(category_path):
        continue

    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((224, 224))  
            new_name = f"img_{count+1}.jpg"
            img.save(os.path.join(combined_dir, new_name), "JPEG")
            count += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

print(f"Cleaned {count} images saved in: {combined_dir}")

