import transformers
import torch
import json
import copy
import argparse
import random

original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/PEAK_counter.json', 'r'))
original_multi_dataset = json.load(open('/home/hmpiao/EasyEdit/data/demo_multi_analysis_all_rephrase.json', 'r'))

multi_dataset = []
print(len(original_dataset))
for j in range(len(original_multi_dataset)):

    for i in range(j, len(original_dataset) - 1):
        print("{}/{} {}/{}".format(i, len(original_dataset), j, len(original_multi_dataset)))
        
        if original_multi_dataset[j]["id"] == original_dataset[i]["case_id"]:
            edit_sample = original_multi_dataset[i]
            edit_sample["rephrase"] = original_dataset[i]["para_add_prompts"]
            edit_sample["locality"] = original_dataset[i]["neighborhood_prompts"]

            multi_dataset.append(edit_sample)

json.dump(multi_dataset, open("/home/hmpiao/EasyEdit/data/demo_multi_analysis_all_rephrase_local.json", 'w'), indent=4)