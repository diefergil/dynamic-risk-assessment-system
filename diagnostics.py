
import pandas as pd
import numpy as np
import timeit
import os
import json
import utils
import subprocess
import sys
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")


##################Function to get model predictions
def model_predictions(dataset_csv_path):
    #read the deployed model and a test dataset, calculate predictions
    X, y = utils.load_data(dataset_csv_path)
    clf = utils.load_model(model_path)
    y_pred = clf.predict(X)
    
    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(dataset_csv_path):
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)
    summary = df.describe().to_string()
    return summary # return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    timing_dict = {'ingestion_time_secs': 0, 'training_time_secs': 0}
    timing_output = []

    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing = timeit.default_timer() - starttime
    timing_dict['ingestion_time_secs'] = round(timing, 2)

    starttime = timeit.default_timer()
    os.system('python training.py')
    timing = timeit.default_timer() - starttime
    timing_dict['training_time_secs'] = round(timing, 2)

    timing_output.append(timing_dict)

    # return a list of 2 timing values in seconds
    return timing_output

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    print(model_predictions(dataset_csv_path))
    print(dataframe_summary(dataset_csv_path))
    print(execution_time())
    print(outdated_packages_list())





    
