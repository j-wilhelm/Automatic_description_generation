import os
import json
import pickle
import time

import tqdm

from h_utils import compute_corpusbleu, compute_ROUGE, diversity_measures

# Additional computation for combined models; need to recalculate corpus bleu for the combination of all predictions.
# Same has to be done for novelty scores etc.

# Configs to use:
config_path = r"C:\Users\s159655\Documents\JADS\Thesis\Code\Train_Test\Configurations"
res_path = r"C:\Users\s159655\Documents\JADS\Thesis\RESULTS\Results_Paper"
# merge_configs = {"RL": "_RL_config_72.json"}
# for i in ["MA", "UO", "AE"]:
#     merge_configs[i] = f"{i}_config_10.json"
#
# parinject_configs = {i: f"{i}_config_403.json" for i in ["MA", "UO", "AE", "RL"]}
# brownlee_configs = {i: f"{i}_config_409.json" for i in ["MA", "UO", "AE", "RL"]}
# for i in ["MA", "UO", "AE", "RL"]:
#     merge_configs = {i: f"{i}_config_501.json" for i in ["MA", "UO", "AE", "RL"]}
# #
#     parinject_configs = {i: f"{i}_config_502.json" for i in ["MA", "UO", "AE", "RL"]}
#     brownlee_configs = {i: f"{i}_config_506.json" for i in ["MA", "UO", "AE", "RL"]}

# Get all train and val descriptions for novelty scores
all_train_descs = []
for w in ["RALPH_LAUREN", "MADEWELL", "American_Eagle", "Urban_Outfitters"]:
    with open(os.path.join(res_path, f"train_descs_{w}.p"), 'rb') as f:
        train_descs = pickle.load(f)
    with open(os.path.join(res_path, f"val_descs_{w}.p"), 'rb') as f:
        val_descs = pickle.load(f)

    train_descs = [" ".join(d) for d in train_descs]
    val_descs = [" ".join(d) for d in val_descs]
    reference_split = train_descs + val_descs

    all_train_descs.extend(reference_split)

# configs_601 = {i: f"{i}_config_601.json" for i in ["MA", "UO", "AE", "RL"]}
# configs_702 = {i: f"{i}_config_702.json" for i in ["MA", "UO", "AE", "RL"]}
configs_411 = {i: f"{i}_config_411.json" for i in ["MA", "UO", "AE"]}
configs_411["RL"] = "RL_config_451.json"
# Get all predictions and references from the configs:
# for model_configs in tqdm.tqdm([merge_configs, parinject_configs, brownlee_configs]):
for model_configs in tqdm.tqdm([configs_411]):
    stime = time.time()
    all_preds = []
    all_refs = []
    results = {}
    print("LOADING DATA ...")
    for webshop, config in tqdm.tqdm(model_configs.items()):
        with open(os.path.join(res_path, f"{webshop.lower()}_results", "all_results",
                               f"results_dict{config}"), "r") as f:
            resdict = json.load(f)
        preds = resdict["PREDICTIONS"]
        references = resdict["REFERENCES"]

        all_preds.extend(preds)
        all_refs.extend(references)

    # Now we have to recompute rouge and bleu scores
    print("COMPUTING ROUGE ...")
    rouge_scores = compute_ROUGE(all_preds, all_refs)
    print("ROUGE: ", rouge_scores)
    results["ROUGE_SCORES"] = {}
    results["ROUGE_SCORES"]["Avg"] = rouge_scores
    # Diversity scores
    print("COMPUTING DIVERSITY MEASURES ...")
    diversity_scores = diversity_measures(all_preds, all_refs, all_train_descs)

    results.update(diversity_scores)

    # bleu computation works by splitting sentence into a list of words
    print("COMPUTING BLEU ...")
    preds = [p.split(" ") for p in all_preds]
    refs = [[r.split(" ") for r in x] for x in all_refs]

    bleu_scores = compute_corpusbleu(refs, preds)
    print("BLEU: ", bleu_scores)
    results["BLEU_SCORES"] = {}
    results["BLEU_SCORES"]["BLEU_1"] = bleu_scores[0]
    results["BLEU_SCORES"]["BLEU_2"] = bleu_scores[1]
    results["BLEU_SCORES"]["BLEU_3"] = bleu_scores[2]
    results["BLEU_SCORES"]["BLEU_4"] = bleu_scores[3]




    # Add preds and references to dict
    results["PREDICTIONS"] = preds
    results["REFERENCES"] = refs

    # Save results dict as JSON FILE
    with open(os.path.join(res_path, "ROBUSTNESS", f"COMBINED_config_411.json"), 'w') as f:
        json.dump(results, f)

    etime = time.time()
    print(f"Computation for these models took {etime - stime} seconds")

# Add novelty to combined configs
# mw_p_res = r"C:\Users\s159655\Documents\JADS\Thesis\RESULTS\Results_Paper\COMBINED"
# for config in os.listdir(mw_p_res):
#     if config not in ["results_dict_combined_config_30.json", "results_dict_combined_config_31.json"]:
#         continue
#     with open(os.path.join(mw_p_res, config), 'r') as f:
#         resdict = json.load(f)
#     preds = resdict["PREDICTIONS"]
#     refs = resdict["REFERENCES"]
#     results_dict = diversity_measures(preds, refs, all_train_descs, verbose=0)
#     print(config, results_dict, resdict["BLEU_SCORES"])
#
#     resdict.update(results_dict)
#
#     with open(os.path.join(mw_p_res, config), 'w') as f:
#         json.dump(resdict, f)
