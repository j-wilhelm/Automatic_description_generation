import os
import pickle
import sys
from functools import partial

import tqdm
import shutil
import multiprocessing as mp

cwd = os.getcwd()

sys.path.append(cwd)
def copy_files(batch, from_f, to_f):
    for i in batch:
        orig_path = os.path.join(from_f, i)
        new_path = os.path.join(to_f, i)
        shutil.copyfile(orig_path, new_path)


from h_dataset_prepping import DataSet_Prepping
webshops = ["RALPH_LAUREN", "MADEWELL", "American_Eagle", "Urban_Outfitters"]

c = "combined"
dpath = os.path.join(cwd, "Datasets", c)
if not os.path.exists(dpath):
    os.mkdir(dpath)
    os.mkdir(os.path.join(dpath, "Descriptions"))
    os.mkdir(os.path.join(dpath, "incv3_noft_GAP_last"))
    os.mkdir(os.path.join(cwd, "variables", c))
    os.mkdir(os.path.join(cwd, "models", "Output", c))
    os.mkdir(os.path.join(cwd, "models", "Weights", c))
    os.mkdir(os.path.join(cwd, "CONFIGS", "co_configs"))

webshop_name_dict = {"RALPH_LAUREN": "RL", "MADEWELL": "MA", "American_Eagle": "AE", "Urban_Outfitters": "UO"}
train_imgs, test_imgs, val_imgs = [], [], []
train_descs, test_descs, val_descs = [], [], []

nr_processors = mp.cpu_count()

# Move files from each webshop to new location
for web in webshops:
    img_path = os.path.join(cwd, "Datasets", web, "incv3_noft_GAP_last")
    desc_path = os.path.join(cwd, "Datasets", web, "Descriptions")
    output_path_img = os.path.join(dpath, "incv3_noft_GAP_last")
    output_path_descs = os.path.join(dpath, "Descriptions")

    all_imgs = os.listdir(img_path)
    all_descs = os.listdir(desc_path)
    batch_size_img = int(len(all_imgs) / nr_processors)
    batch_size_descs = int(len(all_descs) / nr_processors)
    img_batches = [all_imgs[x:x+batch_size_img] for x in range(0, len(all_imgs), batch_size_img)]
    desc_batches = [all_descs[x:x + batch_size_img] for x in range(0, len(all_imgs), batch_size_img)]
    pool = mp.Pool(nr_processors)
    for _ in tqdm.tqdm(pool.imap_unordered(partial(copy_files, from_f=img_path, to_f=output_path_img), img_batches)):
        pass
    for _ in tqdm.tqdm(pool.imap_unordered(partial(copy_files, from_f=desc_path, to_f=output_path_descs), desc_batches)):
        pass
    # Get train, val and test splits:
    main_img_path = os.path.join(cwd, "Datasets", web, "resized_imgs")
    for cat in os.listdir(os.path.join(main_img_path, "TRAIN")):
        for img in os.listdir(os.path.join(main_img_path, "TRAIN", cat)):
            train_imgs.append(img[:-4] + ".p")
            train_descs.append(img[:9] + ".txt")

    for cat in os.listdir(os.path.join(main_img_path, "VAL")):
        for img in os.listdir(os.path.join(main_img_path, "VAL", cat)):
            val_imgs.append(img[:-4] + ".p")
            val_descs.append(img[:9] + ".txt")

    for cat in os.listdir(os.path.join(main_img_path, "TEST")):
        for img in os.listdir(os.path.join(main_img_path, "TEST", cat)):
            test_imgs.append(img[:-4] + ".p")
            test_descs.append(img[:9] + ".txt")

test_descs = set(test_descs)
val_descs = set(val_descs)
train_descs = set(train_descs)
train_imgs = [os.path.join(dpath, "incv3_noft_GAP_last", img) for img in train_imgs]
val_imgs = [os.path.join(dpath, "incv3_noft_GAP_last", img) for img in val_imgs]
test_imgs = [os.path.join(dpath, "incv3_noft_GAP_last", img) for img in test_imgs]
train_descs = [os.path.join(dpath, "Descriptions", desc) for desc in train_descs]
val_descs = [os.path.join(dpath, "Descriptions", desc) for desc in val_descs]
test_descs = [os.path.join(dpath, "Descriptions", desc) for desc in test_descs]

# Save paths to a file for later use
with open(os.path.join(cwd, "variables", c, "train_imgs.p"), 'wb') as f:
    pickle.dump(train_imgs, f)
with open(os.path.join(cwd, "variables", c, "test_imgs.p"), 'wb') as f:
    pickle.dump(test_imgs, f)
with open(os.path.join(cwd, "variables", c, "val_imgs.p"), 'wb') as f:
    pickle.dump(val_imgs, f)\

# Now that all data has been copied, we need to process the new vocabulary; make a DatasetPrepping class for this
dataprep = DataSet_Prepping.__new__(DataSet_Prepping)

dataprep.vocab_options = {"threshold": 5}
dataprep.train_descs = train_descs
dataprep.test_descs = test_descs
dataprep.val_descs = val_descs
dataprep.webshop_name = "combined"

dataprep.build_vocabulary()
dataprep.get_embeddings()
