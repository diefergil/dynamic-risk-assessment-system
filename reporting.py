import json
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from diagnostics import model_predictions
import utils
###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['test_data_path'], "testdata.csv") 
model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
confusion_matrix_path = os.path.join(config["prod_deployment_path"], 'confusionmatrix2.png')


##############Function for reporting
def score_model(dataset_csv_path: str):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    X, y_true = utils.load_data(dataset_csv_path)
    clf = utils.load_model(model_path)
            
    metrics.plot_confusion_matrix(clf, X, y_true, cmap="Blues", colorbar=False)
    plt.savefig(confusion_matrix_path)
    
        



if __name__ == '__main__':
    score_model(dataset_csv_path)
