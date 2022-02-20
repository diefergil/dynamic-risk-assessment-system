import os
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 


prod_deployment_path = config['prod_deployment_path']
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
results_model_path = os.path.join(config['output_model_path'], 'latestscore.txt')
ingested_files_path = os.path.join(
    config['output_folder_path'], 'ingestedfiles.txt')

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
        
    shutil.copy2(model_path, prod_deployment_path)
    shutil.copy2(results_model_path, prod_deployment_path)
    shutil.copy2(ingested_files_path, prod_deployment_path)

if __name__ == "__main__":
    store_model_into_pickle()
