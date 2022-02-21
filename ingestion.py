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

input_files = glob.glob(input_folder_path + "/*.csv")

#############Function for data ingestion


def merge_multiple_dataframe(input_files):
    #check for datasets, compile them together, and write to an output file
    dfs = []
    
    for filename in input_files:
        df_temp = pd.read_csv(filename, index_col=None, header=0)
        print(f"Reading file: {filename}, with {len(df_temp)}")
        dfs.append(df_temp)
    
    df = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates(subset=["corporation"])
    
    return df


if __name__ == '__main__':
    df = merge_multiple_dataframe(input_files)
    print(df)
    df.to_csv(output_folder_path+"/finaldata.csv", index=False)

    with open(output_folder_path+"/ingestedfiles.txt", "w") as f:
        f.write(",".join(input_files))
