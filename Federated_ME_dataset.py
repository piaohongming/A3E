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
    model_kwargs={"torch_dtype": torch.float16, "max_memory": {0: "0GiB", 1: "20GiB", 2: "0GiB", 3: "0GiB", 4: "31GiB", 5: "0GiB", 6: "0GiB", 7: "31GiB"}},
    #device=[0,1],
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

messages_rename_template = [
    {"role": "system", "content": "You are a master of naming."},
    {"role": "user", "content": "Please replace the name {} in the text with a new name (except {}) you design by yourself and have never seen before. Please ensure the new name is not a real celebrity or historical figure. Please ensure the new name is as long as {}. Please output the new name and the text with the new name as a json type {{""new name"": , ""new text"": }} and do not output any other content. The text is: {}"},
]
messages_divide_template = [
    {"role": "system", "content": "You are an expert at splitting texts."},
    {"role": "user", "content": "Please split the text into several sentences according to period. Please just split the text and don't change the sentence. Please output each sentence surrounded by double quotes and seperared by \n. Please do not output any other content. The text is: {}"},
]
messages_validate_template = [
    {"role": "system", "content": "You are a chatbot who knows a lot of celebrities and historical figures."},
    {"role": "user", "content": "Please introduce {} for me. If you could not find any information about {}, please just output: I do not know."},
]
messages_single_question_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The text is a Wikipedia passage about {}. Please ask a question about {} based on the text, so that the answer to the question is in the text. Please ensure the name {} is in the question. Please output your question and the corresponding answer as a json type {{""question"": , ""answer"": }}. Please do not output any other content. The text is: {}"},
]
messages_rephrase_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The dict contains a question about {} and its answer. Please rephrase the question in another nine different ways. Please ensure the answers of the rephrased questions are the same as the previous one. Please ensure {} is in the rephrased questions. Please output each rephrased question surrounded by double quotes and seperared by \n. Please do not output any other content. The dict is: {}"},
]
messages_morename_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The dict contains a question about {} and its answer. Please replace the name {} in the question with nine new names you design by yourself and have never seen before. Please ensure all the new names are not real celebrities or historical figures. Please ensure all the new names are as long as {}. Please first output each new name surrounded by double quotes and seperared by \n. Then output each new question surrounded by double quotes and seperared by \n. Please do not output any other content. The dict is: {}"},
]
messages_morename_moreanswer_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The dict contains a question about {} and its answer. Please replace the name {} in the question with nine new names you design by yourself and have never seen before. Please ensure all the new names are not real celebrities or historical figures. Please ensure all the new names are as long as {}. Please also replace the answer with nine similar but different ones. Please first output each new name surrounded by double quotes and seperared by \n. Then output each new question surrounded by double quotes and seperared by \n. Then output each new answer surrounded by double quotes and seperared by \n. Please do not output any other content. The dict is: {}"},
]
messages_relation_question_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The text contains some entities. You should select an entity from the text and recall another entity (not in the text) with its relation to the selected entity in the text. Then generate a question with the entity selected from the text and the recalled relation. Please ensure that the entity selected from the text is in the question. Please ensure the only one answer of the generated question is the recalled entity (not in the text). Then replace the answer with a similar but different one. Please output as json type {{""selected entity"": , ""recalled entity"": , ""question"": , ""answer"": }}. Please do not output any other content. The text is: {}"},
]
messages_onehop_question_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "I will give you two dicts. The first dict contains a question about {} and its answer {}. The second dict contains a question about {} and its answer. Please generate a question by replacing {} in the question of the first dict with the question of the second dict, and then make this question coherent. Please ensure {} is in the generated question. Please ensure {} is the answer of the generated question. Please output as json type {{""question"": , ""answer"": }}. Please do not output any other content. The first dict is: {}. The second dict is: {}."},
]
messages_multi_analysis_template = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "Please replace the name {} with a new name you design by yourself and have never seen before. Please ensure the new name is as long as {}. Please output the new name as a json type {{\"newname\": }} and do not output any other content."},
]

sd_sc_question_template = "Please introduce {} for me. If you could not find any information about {}, please just output: I do not know."
#"This is a Wikipedia passage about {}: "

original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/demo_multi_analysis.json', 'r'))
sd_sc_dataset = [] #sd冲突测试，同问题补全测试
dc_dataset = [] #不同问题补全测试
ss_dataset = [] #rephrase，ss冲突测试
ds_dataset = [] #问不同的人同一个事情，但答案是一样的，ds冲突测试
dd_dataset = [] #问同一个人不同的事情，问不同的人同一个事情，dd冲突测试
port_dataset = [] #推理测试
multi_analysis_dataset = []
error = 0

print(len(original_dataset))
for i in range(len(original_dataset) - 1):
    if dataset_type == "sd_sc" and original_dataset[i]["concept"] != original_dataset[i + 1]["concept"]:
        print(i, end='\r')
        if i == 100:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        existing_name = concept
        text_to_rename = text + ' ' + labels
        
        messages_rename = copy.deepcopy(messages_rename_template)
        messages_rename[1]["content"] = messages_rename[1]["content"].format(concept, existing_name, concept, text_to_rename)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_rename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        #print(json_rename)

        new_name = json_rename["new name"]
        text_renamed = json_rename["new text"]
        #existing_name = existing_name + ", " + new_name

        #验证
        '''
        messages_validate = copy.deepcopy(messages_validate_template)
        messages_validate[1]["content"] = messages_validate[1]["content"].format(new_name, new_name)
        print(messages_validate[1]["content"])
        outputs = pipeline(
            messages_validate,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][-1]["content"])
        '''
        
        messages_divide = copy.deepcopy(messages_divide_template)
        messages_divide[1]["content"] = messages_divide[1]["content"].format(text_renamed)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_divide,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        sentences_divide = outputs[0]["generated_text"][-1]["content"].split("\n")
        for sentence_index in range(len(sentences_divide)):
            if sentence_index >= 1:
                sentences_divide[sentence_index] = sentences_divide[sentence_index][1:-1]
                sentence = sentences_divide[sentence_index]
                sd_sc_edit_sample = {
                    "text": sd_sc_question_template.format(new_name, new_name),
                    "labels": sentence,
                    "concept": new_name,
                    "type": "incomplete",
                    "id": i
                }
                sd_sc_dataset.append(sd_sc_edit_sample)
        sd_sc_edit_sample = {
            "text": sd_sc_question_template.format(new_name, new_name),
            "labels": ''.join(sentences_divide[1:]),
            "concept": new_name,
            "type": "complete",
            "id": i
        }
        sd_sc_dataset.append(sd_sc_edit_sample)
        json.dump(sd_sc_dataset, open("/home/hmpiao/easyedittrue/data/demo_sd_sc_100.json", 'w'), indent=4)
    elif dataset_type == "dc" and original_dataset[i]["concept"] != original_dataset[i + 1]["concept"]:
        print(i)
        if i == 100:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        existing_name = concept
        text_to_rename = text + ' ' + labels
        
        messages_rename = copy.deepcopy(messages_rename_template)
        messages_rename[1]["content"] = messages_rename[1]["content"].format(concept, existing_name, concept, text_to_rename)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_rename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        #print(json_rename)

        new_name = json_rename["new name"]
        text_renamed = json_rename["new text"]
        #existing_name = existing_name + ", " + new_name
        
        messages_divide = copy.deepcopy(messages_divide_template)
        messages_divide[1]["content"] = messages_divide[1]["content"].format(text_renamed)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_divide,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        sentences_divide = outputs[0]["generated_text"][-1]["content"].split("\n")
        for sentence_index in range(len(sentences_divide)):
            if sentence_index >= 1:
                sentences_divide[sentence_index] = sentences_divide[sentence_index][1:-1]
                sentence = sentences_divide[sentence_index]
                messages_single_question = copy.deepcopy(messages_single_question_template)
                messages_single_question[1]["content"] = messages_single_question[1]["content"].format(new_name, new_name, new_name, sentence)
                #print(messages_divide[1]["content"])
                outputs = pipeline(
                    messages_single_question,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                try:
                    json_single_question = json.loads(outputs[0]["generated_text"][-1]["content"])
                except:
                    continue
                sd_sc_edit_sample = {
                    "text": json_single_question["question"],
                    "labels": json_single_question["answer"],
                    "concept": new_name,
                    "type": "incomplete",
                    "id": i
                }
                sd_sc_dataset.append(sd_sc_edit_sample)
        sd_sc_edit_sample = {
            "text": sd_sc_question_template.format(new_name, new_name),
            "labels": ''.join(sentences_divide[1:]),
            "concept": new_name,
            "type": "complete",
            "id": i
        }
        sd_sc_dataset.append(sd_sc_edit_sample)
        json.dump(sd_sc_dataset, open("/home/hmpiao/easyedittrue/data/demo_dc_100.json", 'w'), indent=4)   
    elif dataset_type == "ss" and original_dataset[i]["concept"] != original_dataset[i + 1]["concept"]:
        print(i)
        if i == 100:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        existing_name = concept
        text_to_rename = text + ' ' + labels
        
        messages_rename = copy.deepcopy(messages_rename_template)
        messages_rename[1]["content"] = messages_rename[1]["content"].format(concept, existing_name, concept, text_to_rename)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_rename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        #print(json_rename)

        new_name = json_rename["new name"]
        text_renamed = json_rename["new text"]
        #existing_name = existing_name + ", " + new_name
        
        messages_divide = copy.deepcopy(messages_divide_template)
        messages_divide[1]["content"] = messages_divide[1]["content"].format(text_renamed)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_divide,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        sentences_divide = outputs[0]["generated_text"][-1]["content"].split("\n")
        sentence = random.sample(sentences_divide[1:], 1)[0][1:-1]
        messages_single_question = copy.deepcopy(messages_single_question_template)
        messages_single_question[1]["content"] = messages_single_question[1]["content"].format(new_name, new_name, new_name, sentence)
        outputs = pipeline(
            messages_single_question,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        try:
            json_single_question = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        single_question = json_single_question["question"]
        single_answer = json_single_question["answer"]
        messages_rephrase = copy.deepcopy(messages_rephrase_template)
        messages_rephrase[1]["content"] = messages_rephrase[1]["content"].format(new_name, new_name, json_single_question)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_rephrase,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        rephrases = outputs[0]["generated_text"][-1]["content"].split("\n")
        for rephrase_index in range(len(rephrases)):
            rephrase = rephrases[rephrase_index][1:-1]
            sd_sc_edit_sample = {
                "text": rephrase,
                "labels": single_answer,
                "concept": new_name,
                "type": "rephrase",
                "id": i
            }
            sd_sc_dataset.append(sd_sc_edit_sample)
        sd_sc_edit_sample = {
            "text": single_question,
            "labels": single_answer,
            "concept": new_name,
            "type": "rephrase",
            "id": i
        }
        sd_sc_dataset.append(sd_sc_edit_sample)
        json.dump(sd_sc_dataset, open("/home/hmpiao/easyedittrue/data/demo_ss_100.json", 'w'), indent=4) 
    elif dataset_type == "ds" and original_dataset[i]["concept"] != original_dataset[i + 1]["concept"]:
        print(i)
        if i == 100:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        existing_name = concept
        text_to_rename = text + ' ' + labels
        
        messages_rename = copy.deepcopy(messages_rename_template)
        messages_rename[1]["content"] = messages_rename[1]["content"].format(concept, existing_name, concept, text_to_rename)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_rename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        #print(json_rename)

        new_name = json_rename["new name"]
        text_renamed = json_rename["new text"]
        #existing_name = existing_name + ", " + new_name
        
        messages_divide = copy.deepcopy(messages_divide_template)
        messages_divide[1]["content"] = messages_divide[1]["content"].format(text_renamed)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_divide,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        sentences_divide = outputs[0]["generated_text"][-1]["content"].split("\n")
        sentence = random.sample(sentences_divide[1:], 1)[0][1:-1]
        messages_single_question = copy.deepcopy(messages_single_question_template)
        messages_single_question[1]["content"] = messages_single_question[1]["content"].format(new_name, new_name, new_name, sentence)
        outputs = pipeline(
            messages_single_question,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        try:
            json_single_question = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        single_question = json_single_question["question"]
        single_answer = json_single_question["answer"]
        messages_morename = copy.deepcopy(messages_morename_template)
        messages_morename[1]["content"] = messages_morename[1]["content"].format(new_name, new_name, new_name, json_single_question)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_morename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        morenames = outputs[0]["generated_text"][-1]["content"].split("\n")
        for morename_index in range(int((len(morenames) - 1) / 2)):
            morename = morenames[morename_index + 1 + int((len(morenames) - 1) / 2)][1:-1]
            sd_sc_edit_sample = {
                "text": morename,
                "labels": single_answer,
                "concept": morenames[morename_index][1:-1],
                "type": "morename",
                "id": i
            }
            sd_sc_dataset.append(sd_sc_edit_sample)
        sd_sc_edit_sample = {
            "text": single_question,
            "labels": single_answer,
            "concept": new_name,
            "type": "morename",
            "id": i
        }
        sd_sc_dataset.append(sd_sc_edit_sample)
        json.dump(sd_sc_dataset, open("/home/hmpiao/easyedittrue/data/demo_ds_100.json", 'w'), indent=4) 
    elif dataset_type == "dd" and original_dataset[i]["concept"] != original_dataset[i + 1]["concept"]:
        print(i)
        if i == 100:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        existing_name = concept
        text_to_rename = text + ' ' + labels
        
        messages_rename = copy.deepcopy(messages_rename_template)
        messages_rename[1]["content"] = messages_rename[1]["content"].format(concept, existing_name, concept, text_to_rename)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_rename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        #print(json_rename)

        new_name = json_rename["new name"]
        text_renamed = json_rename["new text"]
        #existing_name = existing_name + ", " + new_name
        
        messages_divide = copy.deepcopy(messages_divide_template)
        messages_divide[1]["content"] = messages_divide[1]["content"].format(text_renamed)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_divide,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        sentences_divide = outputs[0]["generated_text"][-1]["content"].split("\n")
        sentence = random.sample(sentences_divide[1:], 1)[0][1:-1]
        messages_single_question = copy.deepcopy(messages_single_question_template)
        messages_single_question[1]["content"] = messages_single_question[1]["content"].format(new_name, new_name, new_name, sentence)
        outputs = pipeline(
            messages_single_question,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        try:
            json_single_question = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        single_question = json_single_question["question"]
        single_answer = json_single_question["answer"]
        messages_morename_moreanswer = copy.deepcopy(messages_morename_moreanswer_template)
        messages_morename_moreanswer[1]["content"] = messages_morename_moreanswer[1]["content"].format(new_name, new_name, new_name, json_single_question)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_morename_moreanswer,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][-1]["content"])
        morename_moreanswers = outputs[0]["generated_text"][-1]["content"].split("\n")
        print(morename_moreanswers)
        for morename_moreanswer_index in range(int((len(morename_moreanswers) - 2) / 3)):
            morename_moreanswer = morename_moreanswers[morename_moreanswer_index + 1 + int((len(morename_moreanswers) - 2) / 3)][1:-1]
            sd_sc_edit_sample = {
                "text": morename_moreanswer,
                "labels": morename_moreanswers[morename_moreanswer_index + 2 + int(2 * (len(morename_moreanswers) - 2) / 3)][1:-1],
                "concept": morename_moreanswers[morename_moreanswer_index][1:-1],
                "type": "morename",
                "id": i
            }
            sd_sc_dataset.append(sd_sc_edit_sample)
        sd_sc_edit_sample = {
            "text": single_question,
            "labels": single_answer,
            "concept": new_name,
            "type": "morename",
            "id": i
        }
        sd_sc_dataset.append(sd_sc_edit_sample)
        json.dump(sd_sc_dataset, open("/home/hmpiao/easyedittrue/data/demo_dd_100.json", 'w'), indent=4)         
    elif dataset_type == "port" and original_dataset[i]["concept"] != original_dataset[i + 1]["concept"]:
        print(i)
        if i == 100:
            pass
        text = original_dataset[i]["text"]
        labels = original_dataset[i]["labels"]
        concept = original_dataset[i]["concept"]
        existing_name = concept
        text_to_rename = text + ' ' + labels
        
        messages_rename = copy.deepcopy(messages_rename_template)
        messages_rename[1]["content"] = messages_rename[1]["content"].format(concept, existing_name, concept, text_to_rename)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_rename,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            continue
        #print(json_rename)

        new_name = json_rename["new name"]
        text_renamed = json_rename["new text"]
        #existing_name = existing_name + ", " + new_name
        
        messages_divide = copy.deepcopy(messages_divide_template)
        messages_divide[1]["content"] = messages_divide[1]["content"].format(text_renamed)
        #print(messages_divide[1]["content"])
        outputs = pipeline(
            messages_divide,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1]["content"])
        sentences_divide = outputs[0]["generated_text"][-1]["content"].split("\n")
        for sentence_index in range(len(sentences_divide)):
            if sentence_index >= 1:
                sentences_divide[sentence_index] = sentences_divide[sentence_index][1:-1]
                sentence = sentences_divide[sentence_index]
                messages_single_question = copy.deepcopy(messages_single_question_template)
                messages_single_question[1]["content"] = messages_single_question[1]["content"].format(new_name, new_name, new_name, sentence)
                #print(messages_divide[1]["content"])
                outputs = pipeline(
                    messages_single_question,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                try:
                    json_single_question = json.loads(outputs[0]["generated_text"][-1]["content"])
                except:
                    continue
                single_question = json_single_question["question"]
                single_answer = json_single_question["answer"]
                messages_relation_question = copy.deepcopy(messages_relation_question_template)
                messages_relation_question[1]["content"] = messages_relation_question[1]["content"].format(single_answer)
                #print(messages_divide[1]["content"])
                outputs = pipeline(
                    messages_relation_question,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                try:
                    json_relation_question = json.loads(outputs[0]["generated_text"][-1]["content"])
                except:
                    continue
                relation_selected_entity = json_relation_question["selected entity"]
                relation_recalled_entity = json_relation_question["recalled entity"]
                relation_question = json_relation_question["question"]
                relation_answer = json_relation_question["answer"]
                print(relation_recalled_entity)
                print(relation_answer)
                messages_onehop_question = copy.deepcopy(messages_onehop_question_template)
                messages_onehop_question[1]["content"] = messages_onehop_question[1]["content"].format(relation_selected_entity, relation_answer, new_name, relation_selected_entity, new_name, relation_answer, {"question": relation_question, "answer": relation_answer}, json_single_question)
                #print(messages_divide[1]["content"])
                outputs = pipeline(
                    messages_onehop_question,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                try:
                    json_onehop_question = json.loads(outputs[0]["generated_text"][-1]["content"])
                except:
                    continue
                sd_sc_edit_sample = {
                    "text": json_single_question["question"],
                    "labels": json_single_question["answer"],
                    "concept": new_name,
                    "type": "hop1",
                    "id": i
                }
                sd_sc_dataset.append(sd_sc_edit_sample)
                sd_sc_edit_sample = {
                    "text": json_relation_question["question"],
                    "labels": json_relation_question["answer"],
                    "concept": relation_selected_entity,
                    "type": "hop2",
                    "id": i
                }
                sd_sc_dataset.append(sd_sc_edit_sample)
                sd_sc_edit_sample = {
                    "text": json_onehop_question["question"],
                    "labels": json_onehop_question["answer"],
                    "concept": new_name,
                    "type": "reasoning",
                    "id": i
                }
                sd_sc_dataset.append(sd_sc_edit_sample)
        json.dump(sd_sc_dataset, open("/home/hmpiao/easyedittrue/data/demo_port_100.json", 'w'), indent=4) 
    elif dataset_type == "multi_analysis":
        print("sample {} error {}".format(i, error))
        labels = original_dataset[i]["labels"]
        type = original_dataset[i]["type"]
        if type == "complete":
            multi_analysis_dataset.append(original_dataset[i])
            continue

        messages_multi_analysis = copy.deepcopy(messages_multi_analysis_template)
        messages_multi_analysis[1]["content"] = messages_multi_analysis[1]["content"].format(labels, labels)
        #print(messages_rename[1]["content"])
        outputs = pipeline(
            messages_multi_analysis,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        #print(outputs[0]["generated_text"][-1])
        try:
            json_rename = json.loads(outputs[0]["generated_text"][-1]["content"])
        except:
            error = error + 1
            continue
        #print(json_rename)
        new_name = json_rename["newname"]

        multi_analysis_sample = {
            "text": original_dataset[i]["text"],
            "labels": original_dataset[i]["labels"],
            "new_labels": new_name,
            "concept": original_dataset[i]["concept"],
            "type": original_dataset[i]["type"],
            "id": original_dataset[i]["id"]
        }
        multi_analysis_dataset.append(multi_analysis_sample)
        json.dump(multi_analysis_dataset, open("/home/hmpiao/EasyEdit/data/demo_multi_analysis_edit.json", 'w'), indent=4) 

'''
messages = [
    {"role": "system", "content": "You are a master of naming."},
    {"role": "user", "content": "Please replace the name john russell reynolds in the text with a new name (except Kaidrian Valtor Wystan, Thorold Fothergill Pembroke) you design by yourself and have never seen before. Please ensure the new name is not a real celebrity or historical figure. Please output the new name and the text with the new name as a dict {new name: "", new text:""}. The text is: This is a Wikipedia passage about john russell reynolds. Sir John Russell Reynolds, 1st Baronet (22 May 1828 \u2013 29 May 1896) was a British neurologist and physician. Reynolds was born in Romsey, Hampshire, as the son of John Reynolds, an independent minister, and the grandson of Dr. Henry Revell Reynolds."},
]
'''
'''
messages = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The text is a Wikipedia passage about john russell reynolds. Please ask a question about john russell reynolds based on the text, so that the answer to the question is in the text. Please ensure the name john russell reynolds is in the question. Please output your question and the corresponding answer as a dict {question: , answer: }. The text is: Reynolds was born in Romsey, Hampshire, as the son of John Reynolds, an independent minister, and the grandson of Dr. Henry Revell Reynolds."},
]
'''
'''
messages = [
    {"role": "system", "content": "You are a chatbot who knows a lot of celebrities and historical figures."},
    {"role": "user", "content": "Please introduce John Russell Reynolds for me. If you could not find any information about John Russell Reynolds, please just output: I do not know"},
]
'''
'''
messages = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The dict contains a question about John Russell Reynolds and its answer. Please rephrase the question in another way. Please ensure the answer of the rephrased question is the same as the previous one. Please ensure John Russell Reynolds is in the rephrased question. Please output as a dict {question: , answer: }. The dict is: {question: In which town in Hampshire was John Russell Reynolds born?, answer: Romsey, Hampshire}"},
]
'''
'''
messages = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "The text contains some entities. You should select an entity from the text and recall another entity (not in the text) with its relation to the selected entity in the text. Then generate a question with the entity selected from the text and the recalled relation. Please ensure that the entity selected from the text is in the question. Please ensure the only one answer of the generated question is the recalled entity (not in the text). Please output as two dicts {selected entity: , recalled entity: } and {question: , answer: }. The text is: Romsey, Hampshire"},
]
'''
#one-hop
'''
messages = [
    {"role": "system", "content": "You are an excellent test maker."},
    {"role": "user", "content": "I will give you two dicts. The first dict contains a question about Hampshire and its answer Winchester. The second dict contains a question about john russell reynolds and its answer. Please generate a question by replacing Hampshire in the question of the first dict with the question of the second dict, and then make this question coherent. Please ensure john russell reynolds is in the generated question. Please ensure Winchester is the answer of the generated question. Please output as a dict {question: , answer: }. The first dict is: {question: What is the county town of Hampshire?, answer: Winchester}. The second dict is: {question: In which county was John Russell Reynolds born?, answer: Hampshire}."},
]
'''

#为假名字的替换实体




