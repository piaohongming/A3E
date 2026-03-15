#from googletrans import Translator
import json
import copy
import transformers
import torch
import time
from translate import Translator

'''
translator = Translator(service_urls=[
      'translate.google.com',])
trans=translator.translate('Hello World', src='en', dest='zh-cn')
print(trans.origin)
print(trans.text)
'''
write_path = "/home/hmpiao/EasyEdit/data_relation/new_dataset_2.json" 
data_path_list = ["/home/hmpiao/EasyEdit/data_relation/country-capital.json",
             "/home/hmpiao/EasyEdit/data_relation/country-currency.json",
             "/home/hmpiao/EasyEdit/data_relation/landmark-country.json",
             "/home/hmpiao/EasyEdit/data_relation/national_parks.json",
             "/home/hmpiao/EasyEdit/data_relation/park-country.json",
             "/home/hmpiao/EasyEdit/data_relation/person-instrument.json",
             "/home/hmpiao/EasyEdit/data_relation/person-occupation.json",
             "/home/hmpiao/EasyEdit/data_relation/person-sport.json",
             "/home/hmpiao/EasyEdit/data_relation/product-company.json"]
relation_list = ["capital",
            "currency",
            "country",
            "state",
            "country",
            "musical instruments",
            "occupation",
            "sport",
            "company"]
prompt_template_list = ["The {} of {} is",
                   "The {} used in {} is",
                   "The {} that the {} is located in is",
                   "The {} in America that the {} is located in is",
                   "The {} that the {} is located in is",
                   "The {} used by {} is",
                   "The {} of {} is",
                   "The {} played by {} is",
                   "The {} that designed {} is"]
'''
dest_list = ["su", #sundanese
             "sm", #萨摩亚语
             "sr", #塞尔维亚语
             "sk", #捷克斯洛伐克
             "ko", #韩语
             "de", #德语
             "sn", #shona
             "uk", #ukrainian
            ]
'''

dest_list = ["af", #afrikaans
             "sq", #albanian
             "am", #amharic
             "ar", #arabic
             "hy", #armenian
             "az", #azerbaijani
             "eu", #basque
             "be", #belarusian
            ]

model_id = "/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float32, "max_memory": {0: "0GiB", 1: "31GiB", 2: "0GiB", 3: "0GiB", 4: "31GiB", 5: "0GiB", 6: "0GiB", 7: "31GiB"}},
    #device=[0,1],
    device_map="balanced",
    #max_memory={0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "31GiB", 7: "0GiB"}
)
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


#text, concept, answer
#dataset level origin correct
#dataset_relation level correct 
new_dataset = dict()
origin_correct_dict = dict()
translate_correct_dict = dict()
for relation in relation_list:
    origin_correct_dict[relation] = 0
    for dest in dest_list:
        translate_correct_dict[f"{relation}_{dest}"] = 0
        new_dataset[f"{relation}_{dest}"] = list()

for i in range(len(data_path_list)):
    fr = open(data_path_list[i], "r")
    data = json.load(fr)
    data_index = 0
    print(f"cnmcnmcnm:{len(data)}")
    for d in data:
        origin_prompt = copy.deepcopy(prompt_template_list[i]).format(relation_list[i], d["input"])
        answer = d["output"]

        outputs = pipeline(
            origin_prompt,
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=False,
            #temperature=0.05,
            #top_p=0.9,
        )
        if answer in outputs[0]['generated_text']:
            origin_correct_dict[relation_list[i]] += 1

        p = 0
        while p < len(dest_list):
            #print(f"{i}/{len(dest_list)}")
            lan = dest_list[p]
            
            #translator = Translator()
            #trans=translator.translate(relation_list[i], src='en', dest=lan)
            '''
            try:
                trans=translator.translate(relation_list[i], src='en', dest=lan)
            except:
                print("cnmcnmcnmcnm")
                time.sleep(4)
                continue
            '''
            
            translator=Translator(from_lang="en",to_lang=lan)
            translation = translator.translate(relation_list[i])
            
            
            outputs = pipeline(
                copy.deepcopy(prompt_template_list[i]).format(translation, d["input"]),
                max_new_tokens=10,
                eos_token_id=terminators,
                do_sample=False,
                #temperature=0.05,
                #top_p=0.9,
            )
            if answer in outputs[0]['generated_text']:
                translate_correct_dict[f"{relation_list[i]}_{lan}"] += 1
                p = p + 1
                continue
            
            new_dataset[f"{relation_list[i]}_{lan}"].append(
                {
                    "text": copy.deepcopy(prompt_template_list[i]).format(translation, d["input"]),
                    "concept": d["input"],
                    "answer": d["output"]
                }
            )
            p = p + 1
        data_index += 1
        print(f"relation: {relation_list[i]}; origin: {origin_correct_dict}; trans: {translate_correct_dict}; index: {data_index}")
        json.dump(new_dataset, open(write_path, "w"), indent=4)
        
        


