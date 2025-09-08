# to make multi-channel masks into single channel masks
import numpy as np
#import cupy as np
from PIL import Image
from labelme import utils
import os
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_mod_mask(npa, mask_color_type_1=None, mask_color_type_2=None):
    if npa.ndim == 3:
        mod_img = np.zeros((npa.shape[0], npa.shape[1]), dtype=np.uint8)
        mask_color_type_1 = np.array(mask_color_type_1)
        mask_color_type_2 = np.array(mask_color_type_2)
        mask1 = np.all(npa == mask_color_type_1, axis=-1)
        mask2 = np.all(npa == mask_color_type_2, axis=-1)
        mod_img[mask1] = 1
        mod_img[mask2] = 2
    elif npa.ndim == 2:
        mod_img = npa
    else:
        raise ValueError(f"Image is not 2D or 3D. Please check the image format. Number of dims {npa.ndim}")
    return mod_img
            

def process_masks_multithread(load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2):
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folderpath):
        os.makedirs(save_folderpath)
        print(f"Created directory: {save_folderpath}")
    else:
        print(f"Directory exists: {save_folderpath}")

    # Get the list of all multichannel mask names
    multichannel_mask_names = os.listdir(load_folderpath)
    print(f"There are {len(multichannel_mask_names)} masks to convert. Processing:")

    # Adjust the number of threads to leave some CPU cores available for other processes
    max_workers = max(1, multiprocessing.cpu_count() - 2) # leave 10 cores free

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_mask, filepath, load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2): filepath for filepath in multichannel_mask_names}
        for future in tqdm(as_completed(futures), total=len(futures)):
            filepath = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{filepath} generated an exception: {exc}")

    print(f"Saved masks to {save_folderpath}")

def process_single_mask(filepath, load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2):
    savepath = save_folderpath + filepath
    if os.path.exists(savepath):
        print(f"File {savepath} already exists. Skipping.")
        return

    image = Image.open(load_folderpath + filepath)
    npa = np.array(image)
    mod_img = get_mod_mask(npa, mask_color_type_1, mask_color_type_2)
    utils.lblsave(savepath, mod_img)
    # print("Saved mask to", savepath)

def process_masks(load_folderpath, save_folderpath):
    multichannel_mask_names = os.listdir(load_folderpath)
    
    print("There are "+str(len(multichannel_mask_names))+" to convert. Processing:")
    # load all masks
    for filepath in tqdm(multichannel_mask_names, total=len(multichannel_mask_names)):
        image = Image.open(load_folderpath+filepath)
        npa = np.array(image)
        mod_img = get_mod_mask(npa)
        savepath = save_folderpath+filepath
        utils.lblsave(savepath, mod_img)
    
    print("Saved masks to", save_folderpath)

def printimg(im):
    print("Image shape: ", im.shape)
    print("Image dtype: ", im.dtype)
    print("Dimensions:", im.ndim)
    print("Image min value: ", np.min(im))
    print("Image max value: ", np.max(im))
    print("dimension of each value: ", im[0, 0].shape)
    print("total number of pixels: ", im.size)

    # Check if the image is RGB or BGR
    if im.ndim == 3:  # For RGB/BGR images
        #print("Checking if the image is RGB or BGR...")
        if (im[0, 0, 0] > im[0, 0, 2]):  # Compare the first pixel's Red and Blue channels
            print("The image is likely in BGR format.")
        else:
            print("The image is likely in RGB format.")
    
    # full unique values at each pixel
    if im.ndim == 3:  # For RGB images
        unique_colors = np.unique(im.reshape(-1, im.shape[2]), axis=0)
        print("Unique colors in the image (RGB):")
        print(unique_colors)
    else:  # For single-channel images
        unique_values = np.unique(im)
        print("Unique values in the image (single channel):")
        print(unique_values)
    
    # print some pixels around the center
    # center_x = im.shape[0] // 2
    # center_y = im.shape[1] // 2
    # print("Image center pixel values: ", im[center_x-100: center_x+100, center_y-100: center_y+100])


            
if __name__ == "__main__":
    
    load_folderpath = "/home/snaak/Documents/datasets/bread/BRE_001/og_masks/"
    save_folderpath = "/home/snaak/Documents/datasets/bread/BRE_001/masks/"

    # test pixel values
    # load a random image from load_folderpath
    test_image_name = os.listdir(load_folderpath)[99]
    test_image_name = "image_0000.png"
    test_image_path = load_folderpath + test_image_name
    print("test image path: ", test_image_path)
    test_image = Image.open(test_image_path)
    test_image = np.array(test_image)
    printimg(test_image)

    # mask_color_type_1=[255, 106, 77] # top cheese color - augment first then convert to single channel
    # mask_color_type_2=[250, 250, 55] # other cheese color - augment first then convert to single channel
    
    mask_color_type_1 = [59, 247, 6] # class_1 - convert to single channel first then augment
    mask_color_type_2 = [51, 221, 255] # class_2 - convert to single channel first then augment
    
    

    process_masks_multithread(load_folderpath=load_folderpath, save_folderpath=save_folderpath, mask_color_type_1=mask_color_type_1, mask_color_type_2=mask_color_type_2)   
    