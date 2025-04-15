import os
import shutil
import random
from pathlib import Path

# Define source directories for images and masks
image_dirs = [
    "/home/snaak/Documents/datasets/bologna/bologna_check/augmented_jpg_imgs/",
    "/home/snaak/Documents/datasets/bologna/bologna_check/og_jpg_imgs/",
    "/home/snaak/Documents/datasets/bologna/multiingredient_bologna_kiosk/imgs/",
    "/home/snaak/Documents/datasets/bologna/multiingredient_bologna_kiosk/og_jpg_imgs/",
]

mask_dirs = [
    "/home/snaak/Documents/datasets/bologna/bologna_check/augmented_png_class_masks/",
    "/home/snaak/Documents/datasets/bologna/bologna_check/og_png_class_masks/",
    "/home/snaak/Documents/datasets/bologna/multiingredient_bologna_kiosk/masks/",
    "/home/snaak/Documents/datasets/bologna/multiingredient_bologna_kiosk/og_png_class_masks/",
]

# Define destination directories for the new dataset
output_image_dir = "/home/snaak/Documents/datasets/bologna/bologna_check/imgs"
output_mask_dir = "/home/snaak/Documents/datasets/bologna/bologna_check/masks"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Specify the ratio of samples to select from each dataset
ratios = [
    1.0,  # 100% of dataset1
    1.0,  # 100% of dataset2
    0.059,  # 5.9% of dataset3
    1.0,  # 100% of dataset4
]

# Function to copy randomly selected images and masks with unique prefixes
def create_new_dataset(image_dirs, mask_dirs, output_image_dir, output_mask_dir, ratios):
    for idx, (image_dir, mask_dir) in enumerate(zip(image_dirs, mask_dirs)):
        prefix = f"dataset{idx+1}_"  # Create a unique prefix for each dataset

        # Get the list of files in the directories
        image_files = {Path(f).stem for f in os.listdir(image_dir) if f.endswith(".jpg")}
        mask_files = {Path(f).stem for f in os.listdir(mask_dir) if f.endswith(".png")}

        # Debug: Print the number of files in each directory
        print(f"Dataset {idx+1}: {len(image_files)} images, {len(mask_files)} masks")

        # Find matching files (ignoring extensions)
        matching_files = list(image_files.intersection(mask_files))

        # Debug: Print the matching files
        print(f"Dataset {idx+1}: {len(matching_files)} matching files")

        # Calculate the number of samples based on the ratio
        num_samples = int(len(matching_files) * ratios[idx])

        # Randomly select the specified number of files
        if num_samples < len(matching_files):
            selected_files = random.sample(matching_files, num_samples)
        else:
            selected_files = matching_files  # If fewer files exist, take all

        # Debug: Print the selected files
        print(f"Dataset {idx+1}: {len(selected_files)} files selected")

        for file_stem in selected_files:
            # Add prefix to the filenames
            new_image_name = prefix + file_stem + ".jpg"
            new_mask_name = prefix + file_stem + ".png"

            # Copy image to the output directory
            src_image_path = Path(image_dir) / (file_stem + ".jpg")
            dest_image_path = Path(output_image_dir) / new_image_name

            # Debug: Check if the source image exists
            if not src_image_path.exists():
                print(f"Error: Source image not found: {src_image_path}")
                continue

            shutil.copy(src_image_path, dest_image_path)

            # Copy mask to the output directory
            src_mask_path = Path(mask_dir) / (file_stem + ".png")
            dest_mask_path = Path(output_mask_dir) / new_mask_name

            # Debug: Check if the source mask exists
            if not src_mask_path.exists():
                print(f"Error: Source mask not found: {src_mask_path}")
                continue

            shutil.copy(src_mask_path, dest_mask_path)

            print(f"Copied: {new_image_name} and {new_mask_name}")

# Run the function
create_new_dataset(image_dirs, mask_dirs, output_image_dir, output_mask_dir, ratios)

print("New dataset created successfully!")