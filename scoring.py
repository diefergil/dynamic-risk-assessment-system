import os
from sklearn import metrics
import json
import utils

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
model_path = os.path.join(config["output_model_path"], "trainedmodel.pkl")
score_path = os.path.join(config["output_model_path"], "latestscore.txt")


# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    X, y = utils.load_data(test_data_path)
    clf = utils.load_model(model_path)
    y_pred = clf.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    with open(score_path, "w+") as f:
        f.write(f"f1_score = {str(f1_score)}")


if __name__ == "__main__":
    score_model()
