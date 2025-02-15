import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

'''
There are a few things to note when making your own semantic segmentation dataset:
1. The labelme version I use is 3.16.7. It is recommended to use this version of labelme. Some versions of labelme will cause errors.
   The specific error is: Too many dimensions: 3 > 2
   The installation method is the command line pip install labelme==3.16.7
2. The label map generated here is an 8-bit color map, which is different from the format of the data set in the video.
   Although it looks like a color image, it actually only has 8 bits. At this time, the value of each pixel is the type to which the pixel belongs.
   So it is actually the same format as the VOC data set in the video. Therefore, the data set produced in this way can be used normally. It's also normal.
'''
if __name__ == '__main__':
    jpgs_path = "/home/abhinandan/Desktop/data/Top Layer Grasping/blue/imgs/"
    pngs_path = "/home/abhinandan/Desktop/data/Top Layer Grasping/blue/masks/"
    #classes     = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    classes = ["_background_","top_layer","other_layers"]
    # src_path = "./datasets/blue-235_white-40_mixture"
    # src_path = "./datasets/Demo_20231207/blue_mix_200_test"
    src_path = "/home/abhinandan/Desktop/data/Top Layer Grasping/blue/json"
    
    # count = os.listdir("./datasets/v1.2/")
    count = os.listdir(src_path + "/")
    for i in range(0, len(count)):
        # path = os.path.join("./datasets/v1.2", count[i])
        path = os.path.join(src_path, count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
                
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])

            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)
                
            print(new.shape)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
