# Automatic_description_generation
Code for Thesis titled FashionDesc -  Automatic Description Generation for Fashion Articles Based on Images.

To redo the experiments for my thesis, follow the steps below. Note that this is a process that will take quite a while in total. Especially scraping a full web shop, uploading the data, and finetuning the feature extractors can be time-consuming. You should expect around 4-12 hours for training a model afterwards.

# Running Experiments
1. You can run one of the scrapers by changing the path to the chrome webdriver and automatically extract the necessary data from a web shop. Since website structures change frequently, it may be that you need to adjust the scrapers. 

2. For preprocessing, you should upload the data (IMG and ANNOS) to the folders specified below on the cluster. Also add all scripts to the root folder. Next, you install the requirements.txt file. After this, preprocessing is needed. For this, you need configuration files. Specifically, you can add a set-up.json file to the root folder of your cluster. An example is added in the examples folder. You need to run the m_dataset_prepping.py twice. First with GPU_part = False to run all text-related stuff over multiple CPU processors. Next, with GPU_part = True to finetune feature extractors and extract features. Specifically, you need to fill the following fields: "GPU_part" "webshop_name", "raw_imgs_folder", "raw_anns_folder", "desc_filename_length", "embeddings":, "extractors", "testing". Generally, you want to fill these the way they are set in the example to extract all necessary information. Run an sbatch command with "python m_dataset_prepping.py -j set_up_config.json". 

3. Once the full preprocessing is complete, you can start the experiments. Here, too, you need configuration files. An example is again provided in the example folder (MA_config_1.json). You should run "python m_train_captionmodel.py -j config.json" to train the caption model with the specified configuration files. You can queueu multiple configurations by adding a them to the command with a hyphen (e.g. ma_config_1.json-ma_config_2.json-ma_config_3.json). Once the model is trained, you can evaluate it by running 
"python m_evaluate_model.py -j config.json". This will generate a results dictionary with bleu and rouge scores, as well as the predictions and references. This is stored as results_dictconfig.json in the relevant output folder. You can add the diversity metrics by LOCALLY running "python m_text_diversity.py -j config.json" after you have downloaded the results dict to the correct folder (see below). This will automatically add these metrics to the results dict. You can also queue files for the evaluation and text diversity in the same way as for training.

# Folder Structure
If you are interested in retraining models from the other experiments (robustness, multiple webshop, cropped images), you can take a look at the other m_... scripts in the experiment folder. Moreover, you can look at generate_configurations.json to figure out which additional keys are needed.

Specifically, the scripts expects the following structure locally.

```
.
|-- Results <br>
    |-- w1_results <br>
        |-- all_results <br>
            |-- results_dictw1_config_1.json
    |-- w2_results <br> 
        |-- all_results <br>
|-- Code
    |-- Preprocessing
        |-- h_prep.py
        |-- m_prep.py
    |-- Train_Test
        |-- h_train.py
        |-- h_customGenerator.py
        |-- h_utils.py
        |-- m_train.py
        |-- m_eval.py
    |-- Source Classification
        |-- Source_Classification.ipynb
    |-- Scrapers
        |-- scrape_w1.py
        |-- scrape_w2.py
        |-- scrape_w3.py
```
The structure for the raw data is as follows:
```
.
|-- Data
    |-- W1
        |-- ANNOS
            |-- MEN
                |-- Cat_1
                    |-- item_1.json
                    |-- item_2.json
                    |-- item_3.json
                |-- Cat_2
            |-- WOMEN
                |-- Cat_3
                |-- Cat_4
       |-- IMG
           |-- MEN
                |-- Cat_1
                    |-- item_1_0.jpg
                    |-- item_1_1.jpg
                    |-- item_2_0.json
                |-- Cat_2
            |-- WOMEN
                |-- Cat_3
                |-- Cat_4 
    |-- W1                    
```

The structure on the cluster end is more clearly defined. It is important to follow this structure closely. If so, the main files can be run without manual intervention and all other folders will be automatically added as needed. IMG and ANNOS are structured as above. Folders such as Descriptions and incv3_fromA_GAP_last are added automatically so no need to make those.
```
.
|-- CONFIGS
    |-- A
        |-- a_config_1.json
        |-- a_config_2.json
        |-- a_config_3.json
|-- Datasets
    |-- A
        |-- IMG
        |-- ANNOS
        |-- Descriptions
        |-- incv3_fromA_GAP_last
    |-- B
        |-- IMG
        |-- ANNOS
        |-- Descriptions
        |-- incv3_fromA_GAP_last
    |-- C
        |-- IMG
        |-- ANNOS
        |-- Descriptions
        |-- incv3_fromA_GAP_last
|-- models
    |-- Output
        |-- A
        |-- B
    |-- Weights
        |-- A
        |-- B
|-- variables
    |-- A
    |-- B
|-- h_prep.py
|-- m_prep.py
|-- h_captionModel.py
|-- h_customGenerator.py
|-- h_utils.py
|-- m_train.py
|-- m_eval.py
|-- set_up_webshop_a.json
```
