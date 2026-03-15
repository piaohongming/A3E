import transformers
import torch
import os
import numpy as np
import datetime
import struct
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import hydra
import typing
import copy

class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages):
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]


def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]


def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent


def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value

def scr():
    base_dir = hydra.utils.get_original_cwd()
    if os.path.exists(os.path.join(base_dir,"scr-ssd")):
        scr_dir = os.path.join(base_dir,"scr-ssd")
    else:
        scr_dir = os.path.join(base_dir,"scr")

    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)

    return scr_dir
def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "./ckpts/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")


def get_params(model):
    return model.state_dict()


def get_shape(p, model):
    # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])


def get_logits(x):
    return x.logits if hasattr(x, "logits") else x


def tokenize_gpt(batch, tokenizer, device, test=False, success_answers=None):
    sucess_answers_begin = None
    prompt, label = batch["prompt"], batch["target_new"]
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100  # ignore_index of CrossEntropyLoss
    if test or not label:
        if isinstance(prompt[0], dict):
            tokens = tokenizer.apply_chat_template([Chat([p]) for p in prompt], return_tensors="pt", add_generation_prompt=True, return_dict=True, padding=True, truncation=True)
        else:
            tokens = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    else:
        if isinstance(prompt[0], dict):
            prompt_tokens = tokenizer.apply_chat_template([Chat([p]) for p in prompt], return_tensors="pt", add_generation_prompt=True, return_dict=True, padding=True, truncation=True)
            prompt_ids = prompt_tokens["input_ids"]
            num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
            label_tokens = tokenizer(list(label), return_tensors="pt", padding=True, truncation=True)
            tokens = {}
            tokens["input_ids"] = torch.cat((prompt_tokens["input_ids"], label_tokens["input_ids"]), dim=1)
            tokens['attention_mask'] = torch.cat((prompt_tokens['attention_mask'], label_tokens['attention_mask']), dim=1)
            tokens["labels"] = tokens["input_ids"].clone()
            for i in range(len(prompt)):
                tokens["labels"][i][:num_prompt_toks[i]] = mask_token

            tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
        else:
            #prompt = [prompt[0], prompt[0], prompt[0]]
            #label = [label[0], label[0], label[0]]
            full_prompt = [f"{p} {l}" for p, l in zip(prompt, label)]
            full_prompt_withsuccessanswer = list()
            if len(success_answers) > 0:
                full_prompt_withsuccessanswer = full_prompt
                '''
                for s_a in success_answers:
                    for f_p in full_prompt:
                        full_prompt_withsuccessanswer.append(f"{f_p} {s_a}")
                '''
                '''
                for s_a in success_answers:
                    for p in prompt:
                        full_prompt_withsuccessanswer.append(f"{p} {s_a}")
                '''
            else:
                full_prompt_withsuccessanswer = full_prompt
            print("zxczxczxc")
            
            prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
            num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
            tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
            
            tokens_withsuccessanswer = tokenizer(full_prompt_withsuccessanswer, return_tensors="pt", padding=True, truncation=True)
            
            #print(tokens)
            tokens["labels"] = tokens["input_ids"].clone()
            tokens_withsuccessanswer["labels"] = tokens_withsuccessanswer["input_ids"].clone()
            for i in range(len(prompt)):
                tokens["labels"][i][:num_prompt_toks[i]] = mask_token
            for i in range(len(full_prompt_withsuccessanswer)):
                tokens_withsuccessanswer["labels"][i][:num_prompt_toks[0]] = mask_token
                #tokens['attention_mask'][i][num_prompt_toks[i]:] = 0

            tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token  # What is this doing?
            tokens_withsuccessanswer["labels"][tokens_withsuccessanswer["input_ids"] == tokenizer.pad_token_id] = mask_token

            tokens_clone = copy.deepcopy(tokens)
            tokens_clone_withsuccessanswer = copy.deepcopy(tokens_withsuccessanswer)
            '''
            for i in range(num_prompt_toks[0], len(tokens["labels"][0])):
                tokens["input_ids"] = torch.cat((tokens["input_ids"], tokens_clone["input_ids"].clone()), dim=0)
                tokens["labels"] = torch.cat((tokens["labels"], tokens_clone["labels"].clone()), dim=0)
                tokens["attention_mask"] = torch.cat((tokens["attention_mask"], tokens_clone["attention_mask"].clone()), dim=0)
                tokens["attention_mask"][-1][num_prompt_toks[0]-1:i] = 0
            '''
            for i in range(10):
            #for i in range(num_prompt_toks[0], len(tokens["labels"][0])):
            #for i in range(num_prompt_toks[0], int((len(tokens["labels"][0])+num_prompt_toks[0]+1)/2)-1):
            #for i in range(num_prompt_toks[0], int((len(tokens["labels"][0])+num_prompt_toks[0])/2)):
                tokens["input_ids"] = torch.cat((tokens["input_ids"], tokens_clone["input_ids"].clone()), dim=0)
                tokens["labels"] = torch.cat((tokens["labels"], tokens_clone["labels"].clone()), dim=0)
                tokens["attention_mask"] = torch.cat((tokens["attention_mask"], tokens_clone["attention_mask"].clone()), dim=0)
                #tokens["attention_mask"][-1][num_prompt_toks[0]-1:i] = 0
            '''
            tokens["input_ids"] = torch.cat((tokens["input_ids"], tokens_clone["input_ids"].clone()), dim=0)
            tokens["labels"] = torch.cat((tokens["labels"], tokens_clone["labels"].clone()), dim=0)
            tokens["attention_mask"] = torch.cat((tokens["attention_mask"], tokens_clone["attention_mask"].clone()), dim=0)
            tokens["attention_mask"][-1][0:num_prompt_toks[0]] = 0
            '''
            '''
            #tokens["attention_mask"][0][int((len(tokens["labels"][0])+num_prompt_toks[0]+1)/2)-1:] = 0
            tokens["attention_mask"][0][int((len(tokens["labels"][0])+num_prompt_toks[0])/2):] = 0
            '''
            for i in range(10):
            #for i in range(num_prompt_toks[0], len(tokens["labels"][0])):
            #for i in range(num_prompt_toks[0], int((len(tokens["labels"][0])+num_prompt_toks[0]+1)/2)-1):
            #for i in range(num_prompt_toks[0], int((len(tokens["labels"][0])+num_prompt_toks[0])/2)):
                tokens_withsuccessanswer["input_ids"] = torch.cat((tokens_withsuccessanswer["input_ids"], tokens_clone_withsuccessanswer["input_ids"][0:1].clone()), dim=0)
                tokens_withsuccessanswer["labels"] = torch.cat((tokens_withsuccessanswer["labels"], tokens_clone_withsuccessanswer["labels"][0:1].clone()), dim=0)
                tokens_withsuccessanswer["attention_mask"] = torch.cat((tokens_withsuccessanswer["attention_mask"], tokens_clone_withsuccessanswer["attention_mask"][0:1].clone()), dim=0)
                #for j in range(tokens_clone_withsuccessanswer["input_ids"][0:1].shape[0]):
                    #tokens_withsuccessanswer["attention_mask"][-1-j][num_prompt_toks[0]-1:i] = 0
            '''
            tokens_withsuccessanswer["input_ids"] = torch.cat((tokens_withsuccessanswer["input_ids"], tokens_clone_withsuccessanswer["input_ids"].clone()), dim=0)
            tokens_withsuccessanswer["labels"] = torch.cat((tokens_withsuccessanswer["labels"], tokens_clone_withsuccessanswer["labels"].clone()), dim=0)
            tokens_withsuccessanswer["attention_mask"] = torch.cat((tokens_withsuccessanswer["attention_mask"], tokens_clone_withsuccessanswer["attention_mask"].clone()), dim=0)
            tokens_withsuccessanswer["attention_mask"][-1][0:num_prompt_toks[0]] = 0
            '''
            '''
            for j in range(tokens_clone_withsuccessanswer["input_ids"].shape[0]):
                #tokens_withsuccessanswer["attention_mask"][0+j][int((len(tokens["labels"][0])+num_prompt_toks[0]+1)/2)-1:len(tokens["labels"][0])] = 0
                tokens_withsuccessanswer["attention_mask"][0+j][int((len(tokens["labels"][0])+num_prompt_toks[0])/2):len(tokens["labels"][0])] = 0
            '''
            
    tokens = {f"{k1}": v1.to(device) for k1, v1 in tokens.items()}
    
    tokens_withsuccessanswer = {f"{k1}": v1.to(device) for k1, v1 in tokens_withsuccessanswer.items()}
    return tokens, tokens_withsuccessanswer, len(tokens["labels"][0]), len(success_answers)


def tokenize_qa(batch, tokenizer, device, **kwargs):
    input_sequences, output_sequences = batch["text"], batch["labels"]

    input_encoding = tokenizer(
        list(input_sequences),
        padding="longest",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

    target_encoding = tokenizer(
        list(output_sequences),
        padding="longest",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    )

    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    tokens = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    tokens = {f"{k1}": v1.to(device) for k1, v1 in tokens.items()}
    return tokens

def get_tokenizer(config):
    if config.task == 'hall':
        return tokenize_gpt
    else:
        return tokenize_qa






