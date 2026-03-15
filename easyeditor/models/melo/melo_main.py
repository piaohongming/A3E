from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .melo_hparams import MELOHyperParams
from .melo_multimodal_hparams import MELOMultimodalHyperParams
from .util import get_tokenizer
from .melo import LORA, PATCH
from ...util import nethook


def apply_melo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MELOHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    # only support single edit.we will support sequence edit soon
    if kwargs['edit_type'] == "LORA":    
        if keep_original_weight:
            model=deepcopy(model)
        weights_copy = {}
        device = torch.device(f'cuda:{hparams.device}')
        tokenizer = get_tokenizer(hparams)
        if not isinstance(model, LORA):
            if kwargs['memory'] is not None and len(kwargs['memory']) > 0:
                editor = LORA(model, hparams,tokenizer, torch.cat(kwargs['memory'], dim=0))
            else:
                editor = LORA(model, hparams,tokenizer, None)
        else:
            editor = model
        tokens, tokens_withsuccessanswer, success_answers_begin, num_success_answers = tokenizer(requests[0], tok,device, success_answers=kwargs["success_answers"])
        #print("traintraintrain")
        #print(tokens)
        #print(device)
        #editor.to(device)
        label_tok = editor.edit(tokens, tokens_withsuccessanswer, kwargs["subject_tok"], kwargs["relation_tok"], kwargs["answer_tok"], success_answers_begin, num_success_answers)
        return editor,weights_copy,label_tok
    elif kwargs['edit_type'] == "PATCH":
        if "edit_state" not in kwargs.keys():   
            if keep_original_weight:
                model=deepcopy(model)
            weights_copy = {}
            device = torch.device(f'cuda:{hparams.device}')
            tokenizer = get_tokenizer(hparams)
            if not isinstance(model, PATCH):
                editor = PATCH(model, hparams,tokenizer, torch.cat(kwargs['memory'], dim=0))
            else:
                editor = model
            
            tokens = tokenizer(requests[0], tok,device)
            editor.edit(tokens)
            return editor,weights_copy


def apply_melo_to_multimodal_model(
    model: AutoModelForCausalLM, 
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MELOHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if keep_original_weight:
        model=deepcopy(model)
    weights_copy = {}
    device = torch.device(f'cuda:{hparams.device}')
    if not isinstance(model, LORA):
        editor = LORA(model, hparams,None)  #TODO:多模态lora初始化
    else:
        editor = model
    # Define i/o
    src = [request["prompt"] for request in requests]
    trg = [
        (" " if request["target"][0] != " " else "")
        + request["target"]
        for request in requests
    ]
    image = [request["image"] for request in requests] 
    image = torch.stack(image, dim=0).to(device)
    text_input = [s + t for s, t in zip(src, trg)]
    if hparams.model_name == "minigpt4":
        prompts_len = [len(tok.encode(s, add_special_tokens=False)) for s in src]
        labels = tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"].to(device)
    else:
        prompts_len = [len(tok.encode(s)) for s in src]
        labels = tok(trg, return_tensors="pt",)["input_ids"].to(device)
    
    edit_inner = dict(
        image=image,
        text_input=text_input,
        labels=labels,
        prompts_len=prompts_len
    )
    
    editor.to(device)
    editor.multimodel_edit(edit_inner) 

    return editor,weights_copy