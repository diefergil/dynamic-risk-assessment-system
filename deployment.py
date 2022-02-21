import os
import json
import shutil
import utils
import pickle
##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 


prod_deployment_path = config['prod_deployment_path']
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
results_model_path = os.path.join(config['output_model_path'], 'latestscore.txt')
ingested_files_path = os.path.join(
    config['output_folder_path'], 'ingestedfiles.txt')

####################function for deployment
def store_model_into_pickle(model, ingested_files, score,  path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    pickle.dump(model, open(path+'/trainedmodel.pkl', "wb"))
    
    with open(path+"/ingestedfiles.txt", "w") as f:
        f.write(",".join(ingested_files))
        
    with open(path+"/latestscore.txt", "w") as f:
        f.write(score)

if __name__ == "__main__":
    # read model
    model = utils.load_model(model_path)
    # read ingested files
    ingested_files_path = os.path.join(
        prod_deployment_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'r') as f:
        ingested_files = f.readline()
    # read score  
    previous_score_path = os.path.join(
        prod_deployment_path, 'latestscore.txt')
    with open(previous_score_path, 'r') as f:
        score = f.readline()
    # save in fdeployment folder
    store_model_into_pickle(model, ingested_files, score, prod_deployment_path)
