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

#messages_template = [
    #{"role": "system", "content": "You are a master of naming."},
    #{"role": "user", "content": "Please introduce Edmund Malcolm Fothergill for me. If you could not find any information about Edmund Malcolm Fothergill, please just output: I do not know."},
#]
#messages_template = "The countries that Turkey shares border with are "
#messages_template = "The birth year of Thomas Alva Edison is "
#messages_template = "The death year of Thomas Alva Edison is "
#messages_template = "The birth year, death year and birthplace of Thomas Alva Edison is "
#messages_template = "The continent of the birthplace of Thomas Alva Edison is "
#messages_template = "The name of the head of government of the place of death of Leo Arons is"
#messages_template = "Bertrand Russell is the author of Power: A New Social Analysis, A History of Western Philosophy, "
'''
messages_template = "Q: 'Tim Dorsey, who has written the' A: 'Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill'\n\
    Q: 'Jerusalem, which is the partner town of' A: 'NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan'\n\
        Q: 'Pushkin is the author of' A: 'The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights'\n\
            Q: 'WWE is the owner of' A: 'WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT'\n\
                Q: 'Bertrand Russell is the author of'"
messages_template = "Q: Tim Dorsey, who has written the? A: Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\n\
    Q: Jerusalem, which is the partner town of? A: NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\n\
        Q: Pushkin is the author of? A: The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\n\
            Q: WWE is the owner of? A: WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\n\
                Q: Turkey shares border with?"
'''
messages_template = "The instrument of Tom Fletcher is"

outputs = pipeline(
    messages_template,
    max_new_tokens=20,
    eos_token_id=terminators,
    do_sample=False,
    #temperature=0.05,
    #top_p=0.9,
)
print(outputs)

                




