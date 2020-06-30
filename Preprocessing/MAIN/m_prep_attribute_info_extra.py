import os
import pickle
import sys
from functools import partial

import tqdm

cwd = os.getcwd()
sys.path.append(cwd)

import multiprocessing as mp
import json
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications import inception_v3, resnet50
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np

def crop_move_img_batch(img_batch_ids, cat_dict, img_path_dict, img_cat_dict, cr_path, img_part_dict, img_dict):
    nr_bad_cats = 0
    nr_removed_cats = 0
    for id in img_batch_ids:
        imgres = img_dict[id]["result"]
        original_cat = img_cat_dict[id].split("_")[-1]
        if original_cat not in cat_dict.keys():
            print("CAT:  " + original_cat)
            nr_removed_cats += 1
            continue
        score_high = 0
        for part_res in imgres:
            # Get the correct category; if no category is supplied
            if len(cat_dict[original_cat]) > 1:
                cats_to_compare = cat_dict[original_cat]
            else:
                cats_to_compare = [cat_dict[original_cat]]
            if part_res["category"] in cats_to_compare:
                score_new = part_res["score"]
                if score_new < score_high:
                    continue
                imgcoords = part_res["coordinates"]
                coords = (imgcoords["x_min"], imgcoords["y_min"], imgcoords["x_max"], imgcoords["y_max"])

                # Load img
                img = Image.open(img_path_dict[id])
                # Crop
                img = img.crop(coords)

                # Resize
                img = img.resize((299, 299))

                # Save to correct location
                # New filename
                filename_parts = id[:-4].split("_")
                file_id = id[:-7]
                fname = filenames_dict[file_id] + "_" + filename_parts[-1] + ".JPG"
                new_path = os.path.join(cr_path, img_part_dict[fname], img_cat_dict[id], fname)

                img.save(new_path, format="JPEG")
                score_high = score_new
        if score_high == 0:
            print("+++", id, "---", original_cat, "+++")
            nr_bad_cats += 1
    print(f"COULD NOT FIND {nr_bad_cats} CORRECT IMG CATEGORIES IN THIS BATCH")
    print(f"REMOVED {nr_removed_cats} IMGS IN THIS BATCH AS CAT NOT KNOWN")

def extract_attribute_info(img_batch_ids, cat_dict, img_cat_dict, dataset_path):
    nr_bad_cats = 0
    nr_removed_cats = 0
    for id in img_batch_ids:
        attribute_dict = {}
        imgres = img_dict[id]["result"]
        original_cat = img_cat_dict[id].split("_")[-1]
        if original_cat not in cat_dict.keys():
            print("CAT:  " + original_cat)
            nr_removed_cats += 1
            continue
        score_high = 0
        for part_res in imgres:
            # Get the correct category; if no category is supplied
            if len(cat_dict[original_cat]) > 1:
                cats_to_compare = cat_dict[original_cat]
            else:
                cats_to_compare = [cat_dict[original_cat]]
            if part_res["category"] in cats_to_compare:
                score_new = part_res["score"]
                if score_new < score_high:
                    continue
                attribute_info = part_res["attributes"]
                for att in attribute_info:
                    att_name = att["attribute_type"] + "_" + att["attribute"]
                    attribute_dict[att_name] = att["score"]

                # Save file
                filename_parts = id[:-4].split("_")
                file_id = id[:-7]
                fname = filenames_dict[file_id] + "_" + filename_parts[-1] + ".json"
                with open(os.path.join(dataset_path, "attribute_info", fname), 'w') as f:
                    json.dump(attribute_dict, f)

                score_high = score_new
        if score_high == 0:
            nr_bad_cats += 1
    print(f"COULD NOT FIND {nr_bad_cats} CORRECT IMG CATEGORIES IN THIS BATCH")

def build_model(model_type, nr_classes, mode):
    if model_type == "incv3":
        base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet')
    elif model_type == "resnet50":
        base_model = resnet50.ResNet50(include_top=False, weights='imagenet')
    else:
        raise ValueError("This model type is not supported: {}".format(model_type))
    x = base_model.output
    model = GlobalAveragePooling2D(name="GAP_last")(x)
    if mode == "training":
        model = Dropout(0.5, name="dropout_top")(model)
        model = Dense(2048, activation='relu', name="dense2048_{}{}".format(model_type, mode))(model)
        model = Dense(nr_classes, activation='softmax', name="{}_dense_prediction".format(nr_classes))(model)
        model = Model(inputs=base_model.input, outputs=model)
    elif mode == "extracting":
        model = Model(inputs=base_model.input, outputs=model)
        pass

    return model

def finetune_model(mode, model_type, nr_classes, batch_size, train_folder, val_folder):
    model = build_model(model_type, nr_classes, "training")
    first_trainable_layer_index = model.layers.index(model.get_layer("conv2d_89"))
    for layer in model.layers[:first_trainable_layer_index]:
        layer.trainable = False

    if mode == "fromRL":
        print("Loading RL descriptions")
        weights_path = os.path.join(cwd, "models", "Weights", "RL_classification", "incv3",
                                    "RL_class_incv3_unfr2blocks_epoch146.h5")
        model.load_weights(weights_path, by_name=True)
    elif mode == "fromFF":
        print("Loading FF descriptions")
        weights_path = os.path.join(cwd, "models", "Weights", "FF", "Incv3",
                                    "weights_conv6_01_try1.h5")
        model.load_weights(weights_path, by_name=True)
    else:
        # Freeze everything for first epochs (until ES)
        for layer in model.layers[:-4]:
            layer.trainable = False
        print("Starting from scratch")
    train_generator = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input,
                                         horizontal_flip=True)
    val_generator = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)

    train_gen = train_generator.flow_from_directory(train_folder, batch_size=batch_size,
                                                    target_size=(299, 299), class_mode='categorical')
    val_gen = val_generator.flow_from_directory(val_folder, batch_size=batch_size,
                                                target_size=(299, 299), class_mode='categorical')

    # Add early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

    # Save model weights if improved
    weight_path = os.path.join(cwd, "models", "Weights")
    best_model_folder = os.path.join(weight_path, "feature_extractors")
    if not os.path.isdir(best_model_folder):
        os.mkdir(best_model_folder)
    best_model_path = os.path.join(best_model_folder, f"best_model_{model_type}_bb_{mode}.h5")

    print(model.summary)
    # mc = ModelCheckpoint(best_model_path, monitor='val_loss', mode='min', save_best_only=True)

    # Train model for max 250 epochs; unless from scratch then first train for max 50 then unfreeze and
    # train for more
    results = []
    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples / train_gen.batch_size,
                                  epochs=50, validation_data=val_gen, verbose=1, callbacks=[es])
    results.append(history.history)
    model.save_weights(best_model_path + "_frozen")
    for layer in model.layers[first_trainable_layer_index:]:
        layer.trainable = True

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

    model.compile(optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples / train_gen.batch_size,
                                  epochs=250, validation_data=val_gen, verbose=1, callbacks=[es])
    results.append(history.history)
    model.save_weights(best_model_path)
    # Save progress
    output_path = os.path.join(cwd, "models", "Output", "MADEWELL")
    output_folder = os.path.join(output_path, "feature_extractors")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    with open(os.path.join(output_folder, f"history_{model_type}_bb_{mode}.p"), 'wb') as f:
        pickle.dump(results, f)

    return best_model_path


print("SETTING UP VARIABLES .... ")

cat_dict_ma = {'Bags': ["bag", "backpack"], 'Bandanas': "accessory", 'Belts': "belt", 'Boots': "boots-shoe", 'Hats': "headwear",
               'Jackets': "jacket", 'Jeans': ["pants", "trousers", "jeans"], 'Pants': ["culottes", "jeans", "trousers"], 'Shirts': "shirt", 'Shorts': "shorts",
               'Sneakers': ["ballerinas", "sneakers", "boots-shoe"], 'Socks': "socks", 'Sunglasses': "glasses", 'Sweaters': ["tops", "sweatshirt"],
               'Swim': ["bodysuit", "brassiere", "panties"], 'Tees & Sweatshirts': ["shirt", "tops", "sweatshirt"],
               'Wallets and Accessoires': "purse", 'Wallets and Accessories': ["purse", "accessory"],
               'Bralettes': "brassiere", 'Dresses': ["jumpsuit", "kaftan", "dress"], 'Flats': ["ballerinas", "classic-shoe", "sandals-shoe", "slippers-shoe"],
               'Heels': ["ballerinas" "slippers-shoe", "classic-shoe", "sandals-shoe", "ballerinas"],
               'Jackets & Coats': ["jacket", "coat", "sweatshirt"], 'Mules & Slides': ["sandals-shoe", "slippers-shoe"],
               'Overalls & Jumpsuits': ["dress", "jumpsuit"],
               'Pajamas': ["trousers", "tops", "dress", "shorts", "bodysuits", "jumpsuits"],
               'Sandals': ["slippers-shoe", "sandals-shoe"], 'Skirts': "skirt", 'Slippers': "slippers-shoe",
               'Sweatpants & Leggings': ["sportswear", "sweatpants", "culottes"], 'Sweatshirts': ["tops", "sweatshirt", "jacket", "blazer"], 'Tees': ["tops", "shirt"],
               'Tops & Bras': "tops",
               'Tops & Shirts': ["shirt", "tops", "blouse"], 'Undies': ["brassiere", "panties", "underpants"]}

# Build img_ID --> img_path dict
img_path_dict = {}
img_cat_dict = {}
img_part_dict = {}
dataset_path = os.path.join(cwd, "Datasets", "MADEWELL")
dpath_orig = os.path.join(dataset_path, "IMG")
#
# print(" BUILDING DICTIONARIES ... ")
# for g in os.listdir(dpath_orig):
#     for c in os.listdir(os.path.join(dpath_orig, g)):
#         cpath = os.path.join(dpath_orig, g, c)
#         for img_id in os.listdir(cpath):
#             img_path_dict[img_id] = os.path.join(cpath, img_id)
#             img_cat_dict[img_id] = g + "_" + c
#
# for part in os.listdir(os.path.join(dataset_path, "resized_imgs")):
#     part_path = os.path.join(dataset_path, "resized_imgs", part)
#     for c in os.listdir(part_path):
#         for i in os.listdir(os.path.join(part_path, c)):
#             img_part_dict[i] = part
#
# # Load img dict file
# with open(os.path.join(cwd, "variables", "MADEWELL", "madewell_full_img_resdict.json"), 'r') as f:
#     img_dict = json.load(f)
#
# all_ids = list(img_dict.keys())
#
# # batches
# nr_cpus = mp.cpu_count()
# batch_size = int(len(all_ids) / nr_cpus)
# all_ids_batches = [all_ids[x:x + batch_size] for x in range(0, len(all_ids), batch_size)]
#
# # Make necessary folders in cropped_images
# print("BUILDING FOLDERS ...  ")
cr_path = os.path.join(dataset_path, "cropped_images")
# if not os.path.exists(cr_path):
#     os.mkdir(cr_path)
#     for part in os.listdir(os.path.join(dataset_path, "resized_imgs")):
#         part_path = os.path.join(dataset_path, "resized_imgs", part)
#         os.mkdir(os.path.join(cr_path, part))
#         for cat in os.listdir(part_path):
#             os.mkdir(os.path.join(cr_path, part, cat))
#     os.mkdir(os.path.join(dataset_path, "attribute_info"))
# # Load filename dict
# filename_dict_path = os.path.join(cwd, "variables", "MADEWELL", "filenames_dict.json")
# with open(filename_dict_path, 'r') as f:
#     filenames_dict = json.load(f)
#
# reverse_fn_dict = {value: key for key, value in filenames_dict.items()}
# # Cut images and resize
# print("CROPPING AND RESIZING IMAGES ...  ")
# pool = mp.Pool(nr_cpus)
# for _ in tqdm.tqdm(pool.imap_unordered(partial(crop_move_img_batch, cat_dict=cat_dict_ma, img_cat_dict=img_cat_dict,
#                                                img_part_dict=img_part_dict, img_path_dict=img_path_dict,
#                                                cr_path=cr_path, img_dict=img_dict),
#                                        all_ids_batches), total=len(all_ids_batches)):
#     pass
#
# # Attribute files
# print("BUILDING ATTRIBUTE FILES ...  ")
# pool_2 = mp.Pool(nr_cpus)
# for _ in tqdm.tqdm(pool_2.imap_unordered(partial(extract_attribute_info, cat_dict=cat_dict_ma,
#                                                  img_cat_dict=img_cat_dict, dataset_path=dataset_path),
#                                          all_ids_batches), total=len(all_ids_batches)):
#     pass
# # Attribute dictionary
# print("BUILDING ATTRIBUTES DICTIONARY ...   ")
# attribute_list = []
# attribute_dict = {}
# for att_dict in os.listdir(os.path.join(dataset_path, "attribute_info")):
#     with open(os.path.join(dataset_path, "attribute_info", att_dict), 'r') as f:
#         d = json.load(f)
#
#     for attribute in d.keys():
#         if attribute not in attribute_list:
#             attribute_list.append(attribute)
#
# ix = 1
# for i in attribute_list:
#     attribute_dict[i] = ix
#     ix += 1
#
# # Saving attribute dict
# with open(os.path.join(cwd, "variables", "MADEWELL", "attribute_dict.json"), 'w') as f:
#     json.dump(attribute_dict, f)

# Train feature extractors
# TRAINING EXTRACTORS
train_folder = os.path.join(cr_path, "TRAIN")
val_folder = os.path.join(cr_path, "VAL")
nr_classes = len(os.listdir(train_folder))
model_paths = {}
model_paths["incv3_noft"] = os.path.join(cwd, "models", "Weights", "FF", "Incv3",
                                    "weights_conv6_01_try1.h5")
for mode in ["fromscratch", "fromFF"]:
    for model_type in ["incv3"]:
#         mpath = finetune_model(mode, model_type, nr_classes, 64, train_folder, val_folder)
#         print(mpath)
#         model_paths[f"{model_type}_{mode}"] = mpath
#         K.clear_session()
        model_paths[f"{model_type}_{mode}"] = os.path.join(cwd, "models", "Weights", "feature_extractors",
                                                           f"best_model_incv3_bb_{mode}.h5")

# Extraction
all_imgs = []
for p in os.listdir(cr_path):
    for c in os.listdir(os.path.join(cr_path, p)):
        for i in os.listdir(os.path.join(cr_path, p, c)):
            all_imgs.append(os.path.join(cr_path, p, c, i))

for mode in ["fromscratch", "fromFF", "noft"]:
    model = build_model("incv3", nr_classes, "extracting")
    model.load_weights(model_paths[f"incv3_{mode}"], by_name=True)
    output_path = os.path.join(cwd, "Datasets", "MADEWELL", f"incv3_{mode}_GAP_last_bb")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for img_path in all_imgs:
        img = load_img(img_path, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(img, axis=0)
        x = inception_v3.preprocess_input(x)
        feature_vector = model.predict(x)
        filename = os.path.basename(img_path)[:-4] + ".p"
        with open(os.path.join(output_path, filename), 'wb') as f:
            pickle.dump(feature_vector, f)



    K.clear_session()

# Evaluating Extractors

