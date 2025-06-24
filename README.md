Adapted from DiMB-RE (**Di**et-**M**icro**B**iome dataset for **R**elation **E**xtraction) by Gibong Hong
## Files
Data folder contains:
1. adjudicated.xml - original semrep style file
2. pl-marker_formatted_adjudicated.json - preprocessed file in pl-marker format
3. train.json, dev.json, test.json - train test split for model

Preprocess folder contains:
1. processXMLtoJSON.py - input: semrep style xml file, output: pl-marker style json file
   - There is a slight bug in this file that is fixed by running reformat_json.py
2. create-train-dev-test-split.py - create the splits for model training
3. verification_xml_json.py - verifies that the XML and JSON sentences, entities, and relations correspond

Files that may need to be edited before training:
run_acener_trg_modified.py
run_re_trg_inserted.py

## 1. Setup

### Install dependencies
Please install all the dependency packages using the following command lines to replicate training process, or just use the fine-tuned model:

```bash
conda create -n *your-venv-name* python=3.8
conda activate *your-venv-name*
conda install pip

pip install -r requirements.txt
pip install --editable ./transformers
```

*Note*: We employed and modified the existing codes from [PL-Marker](https://github.com/thunlp/PL-Marker) as a baseline.

## 2. Replicate the Training process for End-to-end RE system

### Training NER and Trigger Extraction model

```bash
bash ./scripts/run_train_ner_PLMarker.sh
```

Check the `run_acener_trg_modified.py` for hyperparameter tuning, code reference, or modifications.
Most of our default parameters follow the original settings suggested by PL-Marker paper.

### Training Relation Extraction model

```bash
bash ./scripts/run_train_re.sh
```

Check the `run_re_trg_inserted.py` for hyperparameter tuning, code reference, or modifications.
Most of our default parameters follow the original settings suggested by PL-Marker paper.

## 3. Details for Pipeline Model

The predictions of the entity model will be saved as a file (`ent_pred_dev.json`) in the `./output` directory if you set `--do_eval`. The predictions (`ent_pred_test.json`) would be generated if you set `--do_test`. The prediction file of the entity model will be the input file of the relation extraction model. This goes same with the relation extraction model: `trg_pred_{dev|test}.json` file would be saved after running the model.

And for evaluation, we recommend you test your prediction file with `run_eval.py` or `run_evals.sh` in order to consider the directionality of predicted relations.