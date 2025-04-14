# dataset augmentor

import albumentations as A
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

transform = A.Compose([
    # A.RandomCrop(height=480, width=480, always_apply=True),
    # A.Resize(height=256, width=256, always_apply=True),
    # A.Blur(blur_limit=(3, 5), p=1.0),
    # A.Sharpen(p=1.0),
    A.RandomBrightnessContrast(always_apply=True),
    # A.VerticalFlip(p=1.0),
    # A.HorizontalFlip(p=1.0),
    # A.RandomRotate90(p=1.0),
])

dataset_path = "/home/snaak/Documents/datasets/cheese/cheese_check/"

# to combine augmentations
# images_dir_name = "augmented_jpg_imgs/"
# masks_dir_name = "augmented_color_masks/"

# # to apply new augmentations
# images_dir_name = "og_jpg_imgs/"
# masks_dir_name = "og_png_color_masks/"

images_dir_name = "spatial_augmented_jpg_imgs/"
masks_dir_name = "spatial_augmented_color_masks/"

# transformed_images_dir_name = "spatial_augmented_jpg_imgs/"
# transformed_masks_dir_name = "spatial_augmented_color_masks/"

transformed_images_dir_name = "intensity_augmented_jpg_imgs/"
transformed_masks_dir_name = "intensity_augmented_color_masks/"


# Set to True if you want to apply the same transformations to masks, keep false for blur / sharpen / brightness / contrast and set to True only for spatial transformations
apply_transform_to_mask = True  

filename_prefix_for_transformed_images_and_masks = "randbc_00"
filetype_for_transformed_images = ".jpg"
filetype_for_transformed_masks = ".png"

# Ensure output directories exist
os.makedirs(dataset_path + transformed_images_dir_name, exist_ok=True)
os.makedirs(dataset_path + transformed_masks_dir_name, exist_ok=True)

# Load images and masks
images = sorted(os.listdir(dataset_path + images_dir_name))
masks = sorted(os.listdir(dataset_path + masks_dir_name))

print(images)
print("_________________________________")
print(masks)
print("Loaded images and masks, beginning augmentation...")

# Thread-safe counter for sequential naming
from threading import Lock
counter_lock = Lock()
counter = 0


def augment_and_save(imagename, maskname):
    global counter
    image = cv2.imread(dataset_path + images_dir_name + imagename)
    mask = cv2.imread(dataset_path + masks_dir_name + maskname)
    transformed = transform(image=image, mask=mask)
    t_image = transformed['image']
    t_mask = transformed['mask'] if apply_transform_to_mask else mask

    # Ensure sequential naming with thread-safe counter
    with counter_lock:
        current_counter = counter
        counter += 1

    img_filepath = dataset_path + transformed_images_dir_name + filename_prefix_for_transformed_images_and_masks + str(current_counter) + filetype_for_transformed_images
    mask_filepath = dataset_path + transformed_masks_dir_name + filename_prefix_for_transformed_images_and_masks + str(current_counter) + filetype_for_transformed_masks

    cv2.imwrite(img_filepath, t_image)
    cv2.imwrite(mask_filepath, t_mask)


# Use ThreadPoolExecutor for multithreading
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(augment_and_save, images, masks), total=len(images)))

print("Augmentation complete!")
print("Saved augmented images to: " + dataset_path + transformed_images_dir_name)
print("Saved augmented masks to: " + dataset_path + transformed_masks_dir_name)