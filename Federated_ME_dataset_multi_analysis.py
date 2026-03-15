import transformers
import torch
import json
import copy
import argparse
import random

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
multi_template = "Tim Dorsey, who has written the Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\
    Jerusalem, which is the partner town of NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\
        Pushkin is the author of The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\
            WWE is the owner of WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\
                {}"

original_dataset = json.load(open('/home/hmpiao/EasyEdit/data/PEAK_time.json', 'r'))
multi_dataset = []
print(len(original_dataset))
for i in range(len(original_dataset) - 1):
    print("{}/{}".format(i, len(original_dataset)))
    concept = original_dataset[i]["requested_rewrite"]["subject"]
    text = original_dataset[i]["requested_rewrite"]["prompt"].format(concept)
    answer = original_dataset[i]["postive_list"]
    rephrase = original_dataset[i]["para_add_prompts"]
    locality = original_dataset[i]["neighborhood_prompts"]
    answer_miss = list()
    answer_success = list()
    outputs = pipeline(
        multi_template.format(text),
        max_new_tokens=50,
        eos_token_id=terminators,
        #do_sample=False,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"])
    for a in answer:
        if a in outputs[0]["generated_text"]:
            answer_success.append(a)
        else:
            answer_miss.append(a)
    
    edit_sample = {
        "text": text,
        "answer": answer,
        "answer_miss": answer_miss,
        "answer_success": answer_success,
        "concept": concept,
        "id": i,
        "rephrase": rephrase,
        "locality": locality
    }
    multi_dataset.append(edit_sample)
    json.dump(multi_dataset, open("/home/hmpiao/EasyEdit/data/demo_multi_analysis_all_rephrase_local_time.json", 'w'), indent=4)

json.dump(multi_dataset, open("/home/hmpiao/EasyEdit/data/demo_multi_analysis_all_rephrase_local_time.json", 'w'), indent=4)

