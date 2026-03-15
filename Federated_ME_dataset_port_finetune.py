import transformers
import torch
import json
import copy
import argparse
import random

original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/demo_port_analysis.json', 'r'))

print(len(original_dataset))
for i in range(len(original_dataset) - 1):
    original_dataset[i]["text"] = original_dataset[i]["text"].replace("is.","is")

json.dump(original_dataset, open("/home/hmpiao/EasyEdit/data/demo_port_analysis.json", 'w'), indent=4)

