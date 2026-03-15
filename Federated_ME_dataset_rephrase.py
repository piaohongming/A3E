import transformers
import torch
import json
import copy
import argparse
import random


model_id = "/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B-Instruct"

#dataset_type = "sd_sc"
#dataset_type = "dc"
#dataset_type = "ss"
#dataset_type = "ds"
#dataset_type = "dd"
#dataset_type = "port"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16, "max_memory": {0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "0GiB", 7: "31GiB"}},
    #device=[0,1],
    device_map="auto"
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/demo_dc_100_rephrase.json', 'r'))
message_statement_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "Please use no more than three words to represent the name of the relation about {} mentioned in the question. Please only output the name of the relation. Please do not output any other content. The question is: {}."},
]
dataset = []
for i in range(len(original_dataset)):
    print(f"{i}/{len(original_dataset)}")
    text = original_dataset[i]["rephrase"]

        
    message_statement = copy.deepcopy(message_statement_template)
    message_statement[1]["content"] = message_statement[1]["content"].format(original_dataset[i]["concept"], text)
    outputs = pipeline(
        message_statement,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    try:
        text_statement = outputs[0]["generated_text"][-1]["content"]
    except:
        print("hhhhhhhhh")
        continue

    #text_statement = json_statement["text"]
    edit_sample = original_dataset[i]
    edit_sample["relation"] = text_statement
    dataset.append(edit_sample)
    json.dump(dataset, open("/home/hmpiao/EasyEdit/data/demo_dc_100_rephrase_2.json", 'w'), indent=4)
    