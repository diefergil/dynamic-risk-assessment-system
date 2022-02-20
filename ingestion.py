import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 
    
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    input_files = glob.glob(input_folder_path + "/*.csv")
    
    dfs = []
    
    for filename in input_files:
        df_temp = pd.read_csv(filename, index_col=None, header=0)
        dfs.append(df_temp)
    
    df = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates(subset=["corporation"])
    
    df.to_csv(output_folder_path+"/finaldata.csv", index=False)
    
    with open(output_folder_path+"/ingestedfiles.txt", "w") as f:
        f.write(",".join(input_files))


if __name__ == '__main__':
    merge_multiple_dataframe()
