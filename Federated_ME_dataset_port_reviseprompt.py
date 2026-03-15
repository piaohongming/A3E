import transformers
import torch
import json
import copy
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', required=True, type=str)
args = parser.parse_args()

model_id = "/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B-Instruct"
dataset_type = args.dataset_type
#dataset_type = "sd_sc"
#dataset_type = "dc"
#dataset_type = "ss"
#dataset_type = "ds"
#dataset_type = "dd"
#dataset_type = "port"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16, "max_memory": {0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "31GiB", 7: "0GiB"}},
    #device=[0,1],
    device_map="auto"
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

if dataset_type == "sd_sc":
    sd_sc_question_template = "The following is an introduction of {}: "
    original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/demo_sd_sc_100.json', 'r'))
    for i in range(len(original_dataset)):
        print(i)
        if i == 10:
            pass
        sd_sc_question = copy.deepcopy(sd_sc_question_template)
        original_dataset[i]['text'] = sd_sc_question.format(original_dataset[i]["concept"])
        json.dump(original_dataset, open("/home/hmpiao/EasyEdit/data/demo_base_sd_sc_100.json", 'w'), indent=4)
else:
    message_statement_template = [
        {"role": "system", "content": "You are an excellent test maker."},
        {"role": "user", "content": "Please help me convert the interrogative tone questions in the text into declarative tone completion questions. For example, change ""When was Amberly Marcellus Waverley born?"" into ""Amberly Marcellus Waverley was born in "". Please do not add a period at the end of the declarative tone completion question. Please add a blank space at the end of the declarative tone completion question. Please output your declarative tone completion question as a json type {{""question"": }}. Please do not output any other content. The text is: {}."},
    ]
    original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/demo_ss_100.json', 'r'))
    dataset = []
    for i in range(len(original_dataset)):
        print(i)
        if i == 10:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        type = original_dataset[i]["type"]
        id = original_dataset[i]["id"]

        if type == "complete":
            sd_sc_question_template = "The following is an introduction of {}: "
            sd_sc_question = copy.deepcopy(sd_sc_question_template)
            text_statement = sd_sc_question.format(original_dataset[i]["concept"])
        else:
            message_statement = copy.deepcopy(message_statement_template)
            message_statement[1]["content"] = message_statement[1]["content"].format(text)
            outputs = pipeline(
                message_statement,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            try:
                json_statement = json.loads(outputs[0]["generated_text"][-1]["content"])
            except:
                continue

            text_statement = json_statement["question"]
        edit_sample = {
            "text": text_statement,
            "labels": labels,
            "concept": concept,
            "type": type,
            "id": id
        }
        dataset.append(edit_sample)
        json.dump(dataset, open("/home/hmpiao/EasyEdit/data/demo_base_ss_100.json", 'w'), indent=4)