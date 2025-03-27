# dataset augmentor

import albumentations as A
import cv2
import os

transform  = A.Compose([
    A.RandomCrop(height=240, width=240, always_apply=True),
    # A.Resize(height=256, width=256, always_apply=True),
    # A.Blur(blur_limit=(3, 5), p=1.0),
    # A.Sharpen(p=1.0),
    # A.RandomBrightnessContrast(always_apply=True),
    # A.VerticalFlip(p=1.0),
    # A.HorizontalFlip(p=1.0),
    # A.RandomRotate90(p=1.0),
])

dataset_path = "/home/snaak/Documents/datasets/cheese/multiingredient_cheese_pickup/"

images_dir_name = "augmented_color_imgs_new/"
masks_dir_name = "augmented_color_masks_new/"

transformed_images_dir_name = "augmented_color_imgs_new/"
transformed_masks_dir_name = "augmented_class_masks_new/"

apply_transform_to_mask = True

filename_prefix_for_transformed_images_and_masks = "randcrop240_00"
filetype_for_transformed_images =".jpg"
filetype_for_transformed_masks = ".png"

transformed_images = []
transformed_masks = []

#load images
images = sorted(os.listdir(dataset_path+images_dir_name))

#load masks
masks = sorted(os.listdir(dataset_path+masks_dir_name))

print(images)
print("_________________________________")
print(masks)
print("Loaded images and masks, beginning augmentation...")

# apply augmentations
for imagename, maskname in zip(images, masks):
    image = cv2.imread(dataset_path+images_dir_name+imagename)
    mask = cv2.imread(dataset_path+masks_dir_name+maskname)
    transformed = transform(image=image, mask=mask)
    t_image = transformed['image']
    transformed_images.append(t_image)
    t_mask = transformed['mask']
    if apply_transform_to_mask:
        transformed_masks.append(t_mask)
    else:
        transformed_masks.append(mask)

print("Augmentation complete!")

counter = 0 #image filename prefix 

# save augmented images and masks
for image, mask in zip(transformed_images, transformed_masks):
    img_filepath = dataset_path+transformed_images_dir_name+filename_prefix_for_transformed_images_and_masks+str(counter)+filetype_for_transformed_images
    cv2.imwrite(img_filepath, image) #TODO create directory if it does not exist; #FORNOW must manually create directory, otherwise images wont be saved
    mask_filepath = dataset_path+transformed_masks_dir_name+filename_prefix_for_transformed_images_and_masks+str(counter)+filetype_for_transformed_masks
    cv2.imwrite(mask_filepath, mask)
    counter=counter+1

print("Saved augmented images to: "+dataset_path+transformed_images_dir_name)
print("Saved augmented masks to: "+dataset_path+transformed_masks_dir_name)




