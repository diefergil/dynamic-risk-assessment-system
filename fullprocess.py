import json
import pickle
import sys
import os
import glob
import subprocess

import training
import scoring
import deployment
import ingestion
import utils
import logging
logger = logging.getLogger()


with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])
input_folder_path = config['input_folder_path']
# Check and read new data
# first, read ingestedfiles.txt
ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
with open(ingested_files_path, 'r') as f:
    ingested_files = f.readline().split(",")

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_input_files = glob.glob(input_folder_path + "/*.csv")
new_input_files = [
    file for file in new_input_files if file not in ingested_files]

if len(new_input_files) > 0:
    logger.info("New data found")
    new_df = ingestion.merge_multiple_dataframe(new_input_files)
    new_df.to_csv(dataset_csv_path+"/finaldata.csv", index=False)

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
    logger.info("Checking current score")
    previous_score_path = os.path.join(
        prod_deployment_path, 'latestscore.txt')

    with open(previous_score_path, 'r') as f:
        current_model_score = float(f.readline().split(" = ")[-1])

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    logger.info("Scoring new model")
    clf = utils.load_model(prod_deployment_path + "/trainedmodel.pkl")
    X, y = utils.splitting_data(new_df)
    new_data_score = scoring.score_model(X, y, clf)
    
# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
    if new_data_score < current_model_score:
        logger.info("There is model drift: Retraining model")
        new_clf = training.train_model(X, y)
        new_model_score = scoring.score_model(X, y, new_clf)
        new_model_score_str = f"f1 score = {new_model_score}"

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
        logger.info("Deployment new model")
        deployment.store_model_into_pickle(
            new_clf, new_input_files, new_model_score_str, "models")

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
        subprocess.call(["python", "reporting.py"])
        subprocess.call(["python", "apicalls.py"])
        
