import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}


# Call each API endpoint and store the responses
response1 = requests.post(
    f"{URL}/prediction?data_path=testdata/testdata.csv", headers=headers).text
response2 = requests.get(f"{URL}/scoring", headers=headers).text
response3 = requests.get(f"{URL}/summarystats", headers=headers).text
response4 = requests.get(f"{URL}/diagnostics", headers=headers).text

# combine all API responses
# responses = #combine reponses here
responses = {
    "predictions": response1,
    'scoring': response2,
    'data_summary': response3,
    'diagnostics': response4
}

# write the responses to your workspace
if __name__ == "__main__":
    
    with open("config.json", "r") as f:
        config = json.load(f)
        
    output_model_path = os.path.join(config.get('output_model_path'), 'apireturns2.txt')
    with open(output_model_path, "w") as f:
        f.write(str(responses))

