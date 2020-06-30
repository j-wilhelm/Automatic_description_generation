# Get all annotation files to get image URLS and product URL. Make a category dictionary. Store full info in csv or JSON.
import json
import os
import time

import tqdm
import requests
import pandas as pd
# # Get all info
# exclude_cats = ["Home", "Gifts & Tech"]
# # poor matches: swim and pajamas
# cat_dict_ma = {'Bags': "bag", 'Bandanas': "[]", 'Belts': "belt", 'Boots': "boots-shoe", 'Hats': "headwear",
#                'Jackets': "jacket", 'Jeans': "jeans", 'Pants': "trousers", 'Shirts': "shirt", 'Shorts': "shorts",
#                'Sneakers': "sneakers", 'Socks': "socks", 'Sunglasses': "glasses", 'Sweaters': "sweatshirt",
#                'Swim': "brassiere", 'Tees & Sweatshirts': "sweatshirt", 'Wallets and Accessoires': "purse",
#                'Wallets and Accessories': "purse",
#                'Bralettes': "brassiere", 'Dresses': "dress", 'Flats': "classic-shoe", 'Heels': "classic-shoe",
#                'Jackets & Coats': "coat", 'Mules & Slides': "slippers-shoe", 'Overalls & Jumpsuits': "jumpsuit",
#                'Pajamas': "bodysuits", 'Sandals': "sandals-shoe", 'Skirts': "skirt", 'Slippers': "slippers-shoe",
#                'Sweatpants & Leggings': "sweatpants", 'Sweatshirts': "sweatshirt", 'Tees': "shirt",
#                'Tops & Bras': "tops",
#                'Tops & Shirts': "tops", 'Undies': "underpants"}
#
# # cat_dict_ae = {"Tops": "tops", }
#
# data_import = []
#
# main_path = r"E:\Jelmer\Uni\Thesis\Data\MADEWELL\ANNOS"
# for gender in os.listdir(main_path):
#     gpath = os.path.join(main_path, gender)
#     for category in tqdm.tqdm(os.listdir(gpath)):
#         cpath = os.path.join(gpath, category)
#         if category in exclude_cats:
#             continue
#         for image in os.listdir(cpath):
#             product_id = image[:-4]
#             # open json file
#             with open(os.path.join(cpath, image), 'r') as f:
#                 annosdict = json.load(f)
#
#             import_dict = {}
#             import_dict["id"] = product_id
#             import_dict["gender"] = gender.lower()
#             import_dict["product_url"] = annosdict["url"]
#             import_dict["product_title"] = annosdict["product_title"]
#             import_dict["category"] = cat_dict_ma[category]
#
#             img_urls = annosdict["img_urls"]
#             if len(img_urls) == 0:
#                 continue
#             for ix_1, url in enumerate(img_urls):
#
#                 # Check whether file exists (for some reason not all images were downloaded)
#                 # img_p = os.path.join(r"E:\Jelmer\Uni\Thesis\Data\MADEWELL\IMG", gender, category,
#                 #                      image[:-4] + f"__{ix_1}" + ".jpg")
#                 # if not os.path.exists(img_p):
#                 #     break
#                 import_dict[f"image_url_{ix_1 + 1}"] = url
#                 if ix_1 == 7:
#                     break
#
#             if (ix_1 + 1) < 8:
#                 for ix_2 in range(ix_1 + 2, 9):
#                     import_dict[f"image_url_{ix_2}"] = ""
#
#             data_import.append(import_dict)
#
# print(len(data_import))
# with open(r"C:\Users\s159655\Documents\JADS\Thesis\Code\Preprocessing\data_import_ma.json", 'w') as f:
#     json.dump(data_import, f)
# #
# with open(r"C:\Users\s159655\Documents\JADS\Thesis\Code\Preprocessing\data_import_ma.txt", 'r') as f:
#     data_import = json.load(f)

#\TODO anonymize this
# Upload
username = ""
password = ""

# # # LOGIN
pixyle_url = "https://pva.pixyle.ai/v2/login"
# data = {"username": "jelmer-wilhelm", "password": "3EnJelme3"}
#
# headers = {"Content-Type": "application/json"}
# r = requests.post(url=pixyle_url, data=json.dumps(data), headers=headers)
# print(r)
# data = r.json()
# print(data)
# # save data if needed in future
# with open("pixyle_login_info.json", 'w') as f:
#     json.dump(data, f)
#
# # Change password
# with open("pixyle_login_info.json", 'r') as f:
#     login_info = json.load(f)
# access_token = "Bearer " + login_info["access_token"]
# print(access_token)
# headers2 = {"Content-Type": "application/json", "Authorization": access_token}
#
# data2 = {"old_password": "zzb", "password": "zzz", "confirm_password": "zzz"}
# pixyle_change_url = "https://pva.pixyle.ai/v2/password/change"
# r2 = requests.post(url=pixyle_change_url, data=json.dumps(data2), headers=headers2)
# print(r2)
# rj = r2.json()
# print(rj)
# with open("pixyle_change_info.json", 'w') as f:
#     json.dump(rj, f)

# # Since JSON is not accepted for some reason
# with open("data_import_ma.json", 'r') as f:
#     data_import = json.load(f)
#
# data_import = pd.DataFrame(data_import)
# data_import.to_csv("data_import_ma.csv", index=False)

# # # Create dataset
# with open("pixyle_login_info.json", 'r') as f:
#     login_info = json.load(f)
# access_token = "Bearer " + login_info["access_token"]
# create_url = "https://pva.pixyle.ai/v2/create"
#
# files = {"file": ("ma.json", open("data_import_ma.json", 'r'), ".json")}
# headers = {"Content_Type": "multipart/form-data", "Authorization": access_token}
# r3 = requests.post(create_url, files=files, headers=headers)
# print(r3)
# d3 = r3.json()
# print(d3)
# with open("pixyle_create_info.json", 'w') as f:
#     json.dump(d3, f)

# Check dataset status
# with open("pixyle_create_info.json", "r") as f:
#     create_info = json.load(f)
# dataset_id = create_info["dataset_id"]
# with open("pixyle_login_info.json", 'r') as f:
#     login_info = json.load(f)
# access_token = "Bearer " + login_info["access_token"]
# xx = requests.get(f"https://pva.pixyle.ai/v2/create/status/{dataset_id}", headers={"Authorization": access_token})
# print(xx)
# print(xx.json())

# Get coordinates
with open("pixyle_create_info.json", "r") as f:
    create_info = json.load(f)
dataset_id = create_info["dataset_id"]
with open("pixyle_login_info.json", 'r') as f:
    login_info = json.load(f)
access_token = "Bearer " + login_info["access_token"]
headers = {"Authorization": access_token}
# # Automated tagging
# tagging_response = requests.get(f"https://pva.pixyle.ai/v2/autotag/{dataset_id}", headers=headers)
# print(tagging_response.json())

# Update
coordinate_json = {}
with open("data_import_ma.json", 'r') as f:
    data_import = json.load(f)
# IDs
ids = []
for i in data_import:
    ids.append(i["id"])

with open("product_info_dict.json", "r") as f:
    coord_json = json.load(f)
headers2 = {"content-type": "application/json", "Authorization": access_token}

#
# for ix, (key, value) in enumerate(coord_json.items()):
#     if ix < 3275:
#         continue
#     try:
#         data = {"dataset_id": dataset_id, "product_id": key, "category": value["result"]["category"],
#                 "attributes": value["result"]["attributes"],
#                 "verified": value["result"]["verified"]}
#     except KeyError:
#         print(ix, key, value)
#         continue
#     r3 = requests.put("https://pva.pixyle.ai/v2/autotag/update", data=json.dumps(data), headers=headers2)
#     r3 = r3.json()
#     if ix % 500 == 0:
#         print(r3)



## Sleep for 2 minutes en check status
# while True:
#     tag_status = requests.get(f"https://pva.pixyle.ai/v2/autotag/status/{dataset_id}", headers=headers)
#     print(tag_status)
#     tag_json = tag_status.json()
#     print(tag_json)
#     if tag_json["meta"]["treated_images_percentage"] == 100.0:
#         break
#     time.sleep(150)

# get coordinates for each product and store in json



# # get information for each id
# coordinate_json = {}
# for iz, id in enumerate(ids):
#     product_info = requests.get(f"https://pva.pixyle.ai/v2/autotag/{dataset_id}/products/{id}", headers=headers)
#     coordinate_json[id] = product_info.json()
#     if iz % 200 == 0:
#         print(coordinate_json)
#     break
# with open("product_info_dict_full.json", "w") as f:
#     json.dump(coord_json, f)
#
# # Get coord file
# r = requests.get(f"https://pva.pixyle.ai/v2/autotag/{dataset_id}/files/json", headers=headers)
# coord_json = r.json()
#
# with open("product_info_dict_full.json", "w") as f:
#     json.dump(coord_json, f)

# Check image coords
# test_url = 'https://i.s-madewell.com/is/image/madewell/AA103_BK5229_d4?wid=500&hei=635&fmt=jpeg&fit=crop&qlt=75,1&resMode=bisharp&op_usm=0.5,1,5,0'
# files = {"image": test_url}
# headers["Content_Type"] = "multipart/form-data"
# r = requests.post("https://pva.pixyle.ai/v2/scale-images", files=files, headers=headers)
# print(r)
# print(r.json())

with open("madewell_full_img_resdict.json", 'r') as f:
    results_dict_images = json.load(f)

men_folder = r"E:\Jelmer\Uni\Thesis\Data\MADEWELL\IMG\MEN"
wo_folder = r"E:\Jelmer\Uni\Thesis\Data\MADEWELL\IMG\WOMEN"
img_info_url = "https://pva.pixyle.ai/v2/autotag/image"
try:
    for x in [men_folder, wo_folder]:
        for cat in tqdm.tqdm(os.listdir(x)):
            for img in tqdm.tqdm(os.listdir(os.path.join(x, cat))):
                img_path = os.path.join(x, cat, img)
                # store imgname\
                img_name = os.path.basename(img)
                if img_name not in results_dict_images.keys():
                    h = {"Authorization": access_token}
                    files = {"image": (img_name, open(img_path, "rb"), "image/jpg")}
                    r = requests.post(img_info_url, files=files, headers=h)
                    try:
                        res = r.json()
                    except json.decoder.JSONDecodeError:
                        print("JSON DECODE ERROR at ", img)
                        continue
                    results_dict_images[img_name] = res

            with open("madewell_full_img_resdict.json", 'w') as f:
                json.dump(results_dict_images, f)
        with open("madewell_full_img_resdict.json", 'w') as f:
            json.dump(results_dict_images, f)
except Exception as e:
    print(e)
    with open("madewell_full_img_resdict.json", 'w') as f:
        json.dump(results_dict_images, f)

with open("madewell_full_img_resdict.json", 'w') as f:
    json.dump(results_dict_images, f)
