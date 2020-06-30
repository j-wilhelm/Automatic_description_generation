import getopt
import os
import sys

dpath = r"C:\Users\s159655\Documents\JADS\Thesis\Code\Train_Test"
sys.path.append(dpath)
import json
import pickle
from h_utils import diversity_measures

def load_data(webshop):
    res_path = r"C:\Users\s159655\Documents\JADS\Thesis\RESULTS\Results_Paper"
    with open(os.path.join(res_path, f"train_descs_{webshop}.p"), 'rb') as f:
        train_descs = pickle.load(f)
    with open(os.path.join(res_path, f"val_descs_{webshop}.p"), 'rb') as f:
        val_descs = pickle.load(f)

    train_descs = [" ".join(d) for d in train_descs]
    val_descs = [" ".join(d) for d in val_descs]
    reference_split = train_descs + val_descs

    return reference_split


arguments = getopt.getopt(sys.argv[1:], shortopts='j:')
print(arguments)

config_files = arguments[0][0][1].split("-")
config_files = [l.strip() for l in config_files]

print(" CONFIGURATIONS:   ", config_files)

webshop_dict = {"AE": "American_Eagle", "UO": "Urban_Outfitters", "RL": "RALPH_LAUREN", "MA": "MADEWELL", "MO": "MONKI"}

# Allows to loop over several configurations so I don't need to monitor this.
for file in config_files:
    res_path = r"C:\Users\s159655\Documents\JADS\Thesis\RESULTS\Results_Paper"
    # res_path_prel = r"C:\Users\s159655\Documents\JADS\Thesis\RESULTS\Results_Preliminary\Prelim_2"
    webshop = file[:2].lower() + "_results"
    filename = f"results_dict{file}"

    try:
        # with open(os.path.join(res_path, webshop, "all_results", filename), 'r') as f:
        #     resdict = json.load(f)
        with open(os.path.join(res_path, webshop, "all_results", filename), 'r') as f:
            resdict = json.load(f)
    except FileNotFoundError:
        # Sometimes the filename is wrong for older files:
        try:
            filename = f"results_dict_{file}"
            with open(os.path.join(res_path, webshop, "all_results", filename), 'r') as f:
                resdict = json.load(f)

        except FileNotFoundError:
            print("Config does not exist ... ", file)
            continue
    webshop_name = webshop_dict[file[:2]]
    reference_partition = load_data(webshop_name)
    # if "novelty_score" in resdict.keys():
    #     print("ALREADY ADDED PARAMETERS TO THIS CONFIG : ", file)
    #     continue
    preds = resdict["PREDICTIONS"]
    refs = resdict["REFERENCES"]
    results_dict = diversity_measures(preds, refs, reference_partition, verbose=0)
    print(file, results_dict, resdict["BLEU_SCORES"])

    resdict.update(results_dict)

    # with open(os.path.join(res_path, webshop, "all_results", filename), 'w') as f:
    #     json.dump(resdict, f)
    with open(os.path.join(res_path, webshop, "all_results", filename), 'w') as f:
        json.dump(resdict, f)