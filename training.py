import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
import utils

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl") 


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    X, y = utils.load_data(dataset_csv_path)
    clf.fit(X, y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(clf, open(model_path, "wb"))

if __name__ == "__main__":
    train_model()
