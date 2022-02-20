from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import json
import os
from diagnostics import dataframe_summary, model_predictions, execution_time



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'])
output_figure_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    Prediction Endpoint.
    Query params: data_path.
    Example:
        curl -X POST '127.0.0.1:8000/prediction?data_path=testdata/testdata.csv'
    Returns:
        - list of model predictions
    """
    data_path = request.args.get("data_path")
    preds = model_predictions(data_path)
  
    #call the prediction function you created in Step 3
    return str(preds), 200 #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    """
    Scoring Endpoint.
    Example:
        curl '127.0.0.1:8000/scoring'
    Returns:
        - check the score of the deployed model in
            `latestscore.txt`
    """
    with open(prod_deployment_path+"/latestscore.txt", "r") as f:
        score = f.readline()
    #check the score of the deployed model
    return score, 200 #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():  
    """
    Stats Endpoint.
        check means, medians, and modes for each column
    Example:
        curl '127.0.0.1:8000/stats'
    Returns:
        - return a list of all calculated summary statistics
    """
    summary = dataframe_summary(dataset_csv_path)
    #check means, medians, and modes for each column
    return summary, 200 #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    diagnostics Endpoint.
        check timing and percent NA values
    Example:
        curl '127.0.0.1:8000/diagnostics'
    Returns:
        - return a json with the exectuion times
    """
    #check timing and percent NA values
    return jsonify(execution_time()), 200 #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
