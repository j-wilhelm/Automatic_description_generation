import os
from PIL import Image
import time
import json
import random


def resize_images_mp(folder_path, filename_dict,
                     output_folder=r"E:\\Jelmer\\Uni\\Thesis\\Data\\Preprocessed\\Images_FF",
                     basewidth=299, baseheight=299):
    # filename_dict = {v: k for k, v in naming_dictionary.items()}

    for image in os.listdir(folder_path):
        if image.endswith(".ini"):
            continue
        # since the image names differ from the desccription names, we need to change this to find the key of the dict
        file_id = image[:-7]
        filename_parts = image[:-4].split("_")
        try:
            new_image_name = filename_dict[file_id] + "_" + filename_parts[-1] + ".JPEG"
        except KeyError:
            print("Could not find the following file:" + str(file_id))
            continue
        if os.path.exists(os.path.join(output_folder, new_image_name)):
            continue

        try:
            im = Image.open(os.path.join(folder_path, image))
        except Exception as e:
            print(e)
            continue
        if im.size[0] < baseheight | im.size[1] < basewidth:
            continue

        im = im.resize((basewidth, baseheight))
        im.save(os.path.join(output_folder, new_image_name), format="JPEG")

        # if nr_images % 500 == 0:
        #     duration = time.time() - start_time
        #     start_time = time.time()
        #     print("Resized 500 images in {} seconds".format(str(duration)))

# with open(r'E:\\Jelmer\\Uni\\Thesis\\Data\\Preprocessed\\filenames_dict_FF.txt', 'r') as f:
#     naming_dictionary = json.load(f)
# filename_dict = {v: k for k,v in naming_dictionary.items()}
# print(len(filename_dict))
