import transformers
import torch
import json
import copy
import argparse
import random

model_id = "/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "max_memory": {0: "0GiB", 1: "20GiB", 2: "0GiB", 3: "0GiB", 4: "31GiB", 5: "0GiB", 6: "0GiB", 7: "31GiB"}},
    #device=[0,1],
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

messages_hop2_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "Here is a text: {}. Please replace '{}' in the text with '{}' and then output the revised text that ends with 'is'. Please do not complete the sentence. Please just output the revised text and do not output any other content."},
]

messages_revise_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "Please output a name which has similar characteristics but different with '{}'. Please output the new name and do not output any other content. Please do not output any other content."},
]

original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/benchmark_wiki_recent_recent_test.json', 'r'))
port_dataset = []
print(len(original_dataset))
for i in range(len(original_dataset) - 1):
    print(i)
    if i >= 10:
        pass
    #hop1
    hop1_concept = original_dataset[i]["subject"]
    hop1_text = original_dataset[i]["prompt"]
    hop1_labels = original_dataset[i]["target_new"]

    #hop2
    if "Reasoning" in original_dataset[i]["portability"].keys():
        reasoning_list = original_dataset[i]["portability"]["Reasoning"]
        for reasoning in reasoning_list:
            hop2_concept = hop1_labels
           
            messages_hop2 = copy.deepcopy(messages_hop2_template)
            if hop1_text.lower() not in reasoning["prompt"].lower():
                continue
            messages_hop2[1]["content"] = messages_hop2[1]["content"].format(reasoning["prompt"], hop1_text, hop1_labels)
            #print(messages_hop2)
            outputs = pipeline(
                messages_hop2,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            #print(outputs[0]["generated_text"][-1]["content"])
            '''
            try:
                hop2_json = json.loads(outputs[0]["generated_text"][-1]["content"])
            except:
                continue
            '''
            hop2_text = outputs[0]["generated_text"][-1]["content"]

            hop2_labels = reasoning["ground_truth"][0][0]
            reasoning_concept = hop1_concept
            reasoning_text = reasoning["prompt"]
            reasoning_labels = reasoning["ground_truth"][0][0]

            

            hop1_edit_sample = {
                "text": hop1_text,
                "labels": hop1_labels,
                "concept": hop1_concept,
                "type": "hop1",
                "id": i
            }
            hop2_edit_sample = {
                "text": hop2_text,
                "labels": hop2_labels,
                "concept": hop2_concept,
                "type": "hop2",
                "id": i
            }
            reasoning_edit_sample = {
                "text": reasoning_text,
                "labels": reasoning_labels,
                "concept": reasoning_concept,
                "type": "reasoning",
                "id": i
            }
            port_dataset.append(hop1_edit_sample)
            port_dataset.append(hop2_edit_sample)
            port_dataset.append(reasoning_edit_sample)
            json.dump(port_dataset, open("/home/hmpiao/EasyEdit/data/demo_port_analysis.json", 'w'), indent=4)
    else:
        continue



