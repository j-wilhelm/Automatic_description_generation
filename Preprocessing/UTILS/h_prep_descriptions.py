import json
import os
from tqdm import tqdm
import string


def process_descriptions_from_path(paths, filenames_dict, path_to_result, foldername, initial_i=0):
    all_paths = []
    redefined_paths = []
    path_result = os.path.join(path_to_result, 'filenames_dict_' + foldername[:-2] + '.txt')
    try:
        i = initial_i
        for path in paths:
            if "MEN" in os.listdir(path) or "WOMEN" in os.listdir(path):
                redefined_paths.append(path)
                for gender in os.listdir(path):
                    p_tmp = os.path.join(path, gender)
                    all_paths.append(p_tmp)

            for category in os.listdir(path):
                full_path = os.path.join(path, category)
                all_paths.append(full_path)

        paths_to_use = set(all_paths) - set(redefined_paths)

        print(all_paths)
        for path_tmp in tqdm(paths_to_use):
            for file in os.listdir(path_tmp):
                if file.endswith('.txt'):
                    i += 1
                    ID_original = file[:-4]

                    # Open item dictionary
                    with open(os.path.join(path_tmp, file), 'r', encoding="utf8") as f:
                        item_dict = json.load(f)

                    # Extract description
                    try:
                        description = item_dict['description']
                        if (description == "") | (description is None):
                            continue
                    except KeyError:
                        continue

                    # Clean description
                    description = description.translate(str.maketrans('', '', string.punctuation))
                    description = description.lower()
                    description = description.replace("<p>", "")
                    description = description.replace("</p>", "")

                    ID_new = str(i).zfill(7)

                    # Get new name and save file
                    filenames_dict[ID_new] = ID_original

                    with open(os.path.join(path_to_result, foldername, ID_new + ".txt"), 'w', encoding='utf8') as fr:
                        fr.write(description)
    except KeyboardInterrupt:
        with open(path_result, 'w') as f:
            json.dump(filenames_dict, f)
    with open(path_result, 'w') as f:
        json.dump(filenames_dict, f)


def get_folders_mp(paths, webshop_name):
    all_paths = []
    redefined_paths = []
    filenames_dict = {}
    i = 0
    basename = webshop_name[:2].upper() + "_"

    for path in paths:
        print(path)
        if "MEN" in os.listdir(path) or "WOMEN" in os.listdir(path):
            redefined_paths.append(path)
            for gender in os.listdir(path):
                p_tmp = os.path.join(path, gender)
                for folder in os.listdir(p_tmp):
                    folderpath = os.path.join(path, gender, folder)
                    all_paths.append(folderpath)

                    for file in os.listdir(os.path.join(path, gender, folder)):
                        if file.endswith(".txt"):
                            i += 1
                            filename_old = file[:-4]
                            filename_new = basename + str(i).zfill(6)
                            filenames_dict[filename_old] = filename_new
        else:
            for category in os.listdir(path):
                full_path = os.path.join(path, category)
                all_paths.append(full_path)

                for file in os.listdir(os.path.join(path, category)):
                    if file.endswith(".txt"):
                        i += 1
                        filename_old = file[:-4]
                        filename_new = basename + str(i).zfill(6)
                        filenames_dict[filename_old] = filename_new

    paths_to_use = set(all_paths) - set(redefined_paths)
    return paths_to_use, filenames_dict


def process_descriptions_from_path_mp(folder_path, output_folder, filenames_dict):
    # output_folder = r"E:\\Jelmer\\Uni\\Thesis\\Data\\Preprocessed\\Prepped_descriptions\\Descriptions_RL"
    # with open(os.path.join(path_to_result, "filenames_dict_RL.txt"), 'r') as f:
    #     filenames_dict = json.load(f)
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            ID_original = file[:-4]

            # Open item dictionary
            with open(os.path.join(folder_path, file), 'r', encoding="utf8") as f:
                item_dict = json.load(f)

            # Extract description
            try:
                description = item_dict['description']
                if (description == "") | (description is None):
                    continue
            except KeyError:
                continue

            # Clean description
            description = description.translate(str.maketrans('', '', string.punctuation))
            description = description.lower()
            description = description.replace("<p>", "")
            description = description.replace("</p>", "")
            description = description.replace("<u>", "")
            description = description.replace("</u>", "")
            description = description.replace("<a>", "")
            description = description.replace("<a", "")
            description = description.replace("</a>", "")

            ID_new = filenames_dict[ID_original]

            with open(os.path.join(output_folder, ID_new + ".txt"), 'w', encoding='utf8') as fr:
                fr.write(description)
