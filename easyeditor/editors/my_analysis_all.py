import os, re, json, sys
sys.path.append("../..")
import torch, numpy
from collections import defaultdict
from easyeditor.util import nethook
from easyeditor.editors import BaseEditor
from easyeditor.models import A3EHyperParams

from easyeditor.editors.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from easyeditor.editors.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import random
import copy

import torch.nn.functional as F
import numpy as np

I = -1
ONLY_ONE_QUESTION = list()
PEN = -1
TEM = -1
LABEL_TOK = list()
BEAM = False

random.seed(43)

last_token_representation = None
last_token_representation_resubmit = None
last_token_representation_gen = None
x_before = None

t_global = None
t_global_change = None
t_global_hopping = None
t_global_change_hopping = None

multi_results = dict()
multi_results["pre"] = dict()
multi_results["post"] = dict()
multi_results["pre"]["origin"] = dict()
multi_results["pre"]["hopping"] = dict()
multi_results["post"]["origin"] = dict()
multi_results["post"]["hopping"] = dict()
multi_results["pre"]["origin"]["prob"] = dict()
multi_results["pre"]["origin"]["rank"] = dict()
multi_results["pre"]["hopping"]["prob"] = dict()
multi_results["pre"]["hopping"]["rank"] = dict()
multi_results["post"]["origin"]["prob"] = dict()
multi_results["post"]["origin"]["rank"] = dict()
multi_results["post"]["hopping"]["prob"] = dict()
multi_results["post"]["hopping"]["rank"] = dict()
multi_results["pre"]["generate"] = list()
multi_results["post"]["generate"] = list()

multi_template = "Tim Dorsey, who has written the Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\
    Jerusalem, which is the partner town of NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\
        Pushkin is the author of The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\
            WWE is the owner of WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\
                {}"
multi_template_rephrase = "Tim Dorsey, who has written the Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\
    Jerusalem, which is the partner town of NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\
        Pushkin is the author of The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\
            WWE is the owner of WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\
                {}"

sequential_template_edit = "{}"
sequential_template = "The answers of the questions \"Daimler has made the\", \"Saxony is adjacent to\" are Mercedes-Benz, Hamburg respectively.\n\
    The answers of the questions \"Karlheinz Stockhausen is the composer of\", \"Daimler is the parent organization of\" are Originale, Car2go respectively.\n\
        The answers of the questions \"Katherine Roberts is the writer of\", \"Florida International University has the employer\" are The Colossus Crisis, Carlos Alvarez respectively.\n\
            The answers of the questions \"contraception has a subclass of\", \"Kering has subsidiary\" are Intrauterine device, Volcom respectively\n\
                The answers of the questions \"{}\", \"{}\" are"

sequential_template_search = "The answers of the questions \"{}\", \"{}\" are"


def find_sublist_position(main_list, sub_list, prompt):
    position = 10000
    position_before = 10000
    for i, element in enumerate(main_list):
        if element == sub_list[0] and sub_list == main_list[i:i+len(sub_list)] and i >= len(prompt):
            position = i+len(sub_list)-1
            position_before = i
            return position, position_before
    return position, position_before

def find_sublist_position_inprompt(prompt, sub_list):
    position = 10000
    position_before = 10000
    for i, element in enumerate(prompt):
        if element == sub_list[0] and sub_list == prompt[i:i+len(sub_list)]:
            position = i+len(sub_list)-1
            position_before = i
            return position, position_before
    return position, position_before

def find_sublist_position_context(prompt, sub_list):
    position = 10000
    position_before = 10000
    for i, element in enumerate(prompt):
        if element == sub_list[0] and sub_list == prompt[i:i+len(sub_list)]:
            position = i+len(sub_list)-1
            position_before = i
    return position, position_before

def get_tgt_tok_id(prefix):
    inner_prefix = make_inputs(mt.tokenizer, [prefix])
    inner_toks_prefix = [mt.tokenizer.decode(inner_prefix["input_ids"][0][i]) for i in range(inner_prefix["input_ids"].shape[1])]
    return len(inner_toks_prefix) - 1

def enc_tok(check_tok, avg=False):
        if avg == False:
            check_tok_enc = mt.tokenizer.encode(check_tok)[1]
        else:
            check_tok_enc = mt.tokenizer.encode(check_tok)[1:]
        if isinstance(check_tok_enc, list):
            return check_tok_enc
        elif isinstance(check_tok_enc, int):
            return [check_tok_enc]
        else:
            raise Exception("format is not expected")

def get_rank(logits, check_tok_enc):
                    logits_dict = dict()
                    for i in range(len(logits)):
                        logits_dict[i] = logits[i]
                    logits_dict = sorted(logits_dict.items(),key=lambda item:item[1], reverse=True)

                    cnt = 0

                    temp_rank = dict()
                    for enc in check_tok_enc:
                        temp_rank[enc] = 0

                    for elem in logits_dict:
                        cnt += 1
                        key, value = elem
                        if key in check_tok_enc:
                            temp_rank[key] += cnt

                    temp_rank_sum = sum([v for v in temp_rank.values()])
                    return temp_rank_sum/len(check_tok_enc)


def trace_with_patch_list(
    model,
    inp,
    inp_rephrase,
    states_to_patch,
    answer_toks_list,
    answer_list,
    prompt,
    prompt_rephrase,
    state,
    addi,
    addi_tok,
    other_model=None,
    success_answers=None,
    false_answers=None,
    inp_for_search=None
    ):
    global I
    global ONLY_ONE_QUESTION
    global LABEL_TOK
    success_prob = 0
    false_prob = 0

    false_answers = [LABEL_TOK[I]]

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    x_before_list = ["model.layers.14"]


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep(x, layer):
        global t_global
        global t_global_change
        global x_before

        if layer not in patch_spec:
            return x

        x_copy = copy.deepcopy(x)
        x_copy = untuple(x_copy)
        norm_x = Norm(x_copy.to(Norm_W.device))
        h = untuple(norm_x)
        if t_global is None:
            t_global = h.shape[1]-1
            t_global_temp = t_global
            t_global_change = 0
        else:
            t_global_change = t_global_change + 1
            t_global_temp = t_global + int(t_global_change // 32)

        t = h.shape[1]-1
        out = h[0, t].to(W.device)
        v = torch.matmul(W, out)
        if len(v.shape) == 1:
            v = v.unsqueeze(1)

        v = v.squeeze()
        logits = torch.softmax(v, dim=0).tolist()

        if state == "pre":
            answer_list_1 = copy.deepcopy(addi)
            answer_list_1.extend(answer_list)
            answer_toks_list_1 = copy.deepcopy(addi_tok)
            answer_toks_list_1.extend(answer_toks_list)
        else:
            answer_list_1 = answer_list
            answer_toks_list_1 = answer_toks_list

        for answer in answer_toks_list_1:

            prob = 0
            for tok in answer:
                prob += logits[tok]
            prob /= len(answer)
            rank = get_rank(logits, answer)

            answer_string = answer_list_1[answer_toks_list_1.index(answer)]
            if answer_string not in multi_results[state]["origin"]["prob"].keys():
                multi_results[state]["origin"]["prob"][answer_string] = dict()
                multi_results[state]["origin"]["rank"][answer_string] = dict()

            if str(t_global_temp) in multi_results[state]["origin"]["prob"][answer_string].keys():
                multi_results[state]["origin"]["prob"][answer_string][str(t_global_temp)].append(prob)
                multi_results[state]["origin"]["rank"][answer_string][str(t_global_temp)].append(rank)
            else:
                multi_results[state]["origin"]["prob"][answer_string][str(t_global_temp)] = []
                multi_results[state]["origin"]["rank"][answer_string][str(t_global_temp)] = []
                multi_results[state]["origin"]["prob"][answer_string][str(t_global_temp)].append(prob)
                multi_results[state]["origin"]["rank"][answer_string][str(t_global_temp)].append(rank)

        return x

    def patch_use_representation(x, layer):
        if layer in ["model.layers.5.mlp.down_proj"]:
            other_model.model.use_representation(x)
        return x

    maxtoks = max([len(p) for p in false_answers+success_answers])

    if state == "pre":
        other_model.model.set_beam(True, I, ONLY_ONE_QUESTION)

        outputs_exp_temp = model.generate(
            input_ids=inp['input_ids'],
            attention_mask=inp['attention_mask'],
            max_new_tokens=15,
            do_sample=True,
            temperature=TEM,
            repetition_penalty=PEN,
            num_beams=1,
        )

        other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)

        gen_text_temp = [mt.tokenizer.decode(x_temp, skip_special_tokens=True) for x_temp in outputs_exp_temp.detach().cpu().numpy().tolist()[0]]
        gen_text_1_temp = ''
        char_tok_dict_temp = dict()
        j = 0
        for t in range(len(gen_text_temp)):
            t_1 = gen_text_temp[t]
            for i in range(len(t_1)):
                gen_text_1_temp = gen_text_1_temp + t_1[i]
                char_tok_dict_temp[j] = t
                j = j + 1

        answer_toks_list_new = []
        answer_list_new = []
        answer_toks_position_list_new = []

        for i in range(len(answer_list)):
            answer = answer_list[i]
            position, position_before = find_sublist_position(gen_text_1_temp, answer, prompt)
            if position != 10000:
                toks = char_tok_dict_temp[position]
                toks_before = char_tok_dict_temp[position_before]
                answer_toks = outputs_exp_temp.detach().cpu().numpy().tolist()[0][toks_before:toks+1]
                answer_toks_list_new.append(answer_toks)
                answer_list_new.append(answer)
                answer_toks_position_list_new.append([toks_before, toks])

        answer_toks_list = answer_toks_list_new
        answer_list = answer_list_new

    if state != "pre":
        other_model.model.set_list()
        other_model.model.set_attention()
        other_model.model.get_submodule(other_model.lora_list[0]).set_stable_tok()

    # Search phase
    with nethook.TraceDict(
        model,
        ["model.layers.5.mlp.down_proj"],
        edit_output=patch_use_representation,
    ) as td:
        other_model.model.set_search(True)
        other_model.model.set_beam(False, I, ONLY_ONE_QUESTION)
        if BEAM:
            other_model.model.set_beam(True, I, ONLY_ONE_QUESTION)
            outputs_exp = model.generate(
                    input_ids=inp_for_search['input_ids'],
                    attention_mask=inp_for_search['attention_mask'],
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=TEM,
                    repetition_penalty=PEN,
                    num_beams=1,
                )
            other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)
        else:
            outputs_exp, _, _, _ = generate_with_cache(
                model,
                prompt_tokens=inp_for_search['input_ids'],
                max_gen_len=1,
                attention_mask=inp_for_search['attention_mask'],
                other_model = other_model,
                optimization=True,
                success_answers=success_answers,
                false_answers=false_answers
            )
        other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)
        other_model.model.set_search(False)

    global t_global
    global t_global_change
    t_global = None
    t_global_change = None

    if state != "pre":
        other_model.model.set_list()
        other_model.model.set_attention()
        other_model.model.get_submodule(other_model.lora_list[0]).set_stable_tok()

    # Generation phase
    with nethook.TraceDict(
        model,
        ["model.layers.5.mlp.down_proj"],
        edit_output=patch_use_representation,
    ) as td:
        other_model.model.set_beam(False, I, ONLY_ONE_QUESTION)
        if BEAM:
            other_model.model.set_beam(True, I, ONLY_ONE_QUESTION)
            outputs_exp = model.generate(
                    input_ids=inp['input_ids'],
                    attention_mask=inp['attention_mask'],
                    max_new_tokens=15,
                    do_sample=True,
                    temperature=TEM,
                    repetition_penalty=PEN,
                    num_beams=1,
                )
            other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)
        else:
            outputs_exp, _, success_prob, false_prob = generate_with_cache(
                model,
                prompt_tokens=inp['input_ids'],
                max_gen_len=15,
                attention_mask=inp['attention_mask'],
                other_model = other_model,
                optimization=True,
                success_answers=success_answers,
                false_answers=false_answers
            )
        other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)

    t_global = None
    t_global_change = None

    if state != "pre":
        other_model.model.set_list()
        other_model.model.set_attention()
        other_model.model.get_submodule(other_model.lora_list[0]).set_stable_tok()

    # Attention map phase (currently pass-through)
    with nethook.TraceDict(
        model,
        ["model.layers.5.mlp.down_proj"],
        edit_output=patch_use_representation,
    ) as td:
        other_model.model.set_beam(False, I, ONLY_ONE_QUESTION)
        if BEAM:
            other_model.model.set_beam(True, I, ONLY_ONE_QUESTION)
            other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)
        else:
            pass
        other_model.model.set_beam(False, -1, ONLY_ONE_QUESTION)

    if state != "pre":
        other_model.model.set_list()
        other_model.model.set_attention()
        other_model.model.get_submodule(other_model.lora_list[0]).set_stable_tok()

    outputs_rephrase = []
    outputs_for_save = []
    outputs_for_resubmit = []
    outputs_for_resubmit_resubmit = []
    outputs_for_resubmit_gen = []
    if state == "pre":
        return outputs_exp, outputs_rephrase, answer_toks_list, answer_list, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen, None, answer_toks_position_list_new
    else:
        return outputs_exp, outputs_rephrase, answer_toks_list, answer_list, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen, None, success_prob, false_prob


def generate_with_cache(
    model,
    prompt_tokens,
    max_gen_len,
    attention_mask,
    other_model,
    optimization,
    success_answers,
    false_answers
):
        global PEN

        one = True
        kvcache = None
        cache_list = []
        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = max_gen_len + max_prompt_len
        pad_id = mt.tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        attention_mask_all = torch.full((bsz, total_len), 1, dtype=torch.long, device="cuda")
        for k, t in enumerate(attention_mask):
            attention_mask_all[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            out = model.forward(tokens, prev_pos, output_attentions=True, attention_mask=attention_mask_all)
            cache_list.append(out.attentions)

        stop_tokens = torch.tensor(list([mt.tokenizer.eos_token_id]), device="cuda")

        need_train = 0
        prob_list_list = list()
        prob_list_list_false = list()
        for a in range(len(success_answers)):
            prob_list_list.append(list())
        for a in range(len(false_answers)):
            prob_list_list_false.append(list())
        for cur_pos in range(min_prompt_len-1, total_len):
            if other_model is not None:
                attention_mask_list = other_model.model.get_attention()
                if attention_mask_list is not None:
                    attention_mask_all[:, attention_mask_list[0]:attention_mask_list[1]] = 0
            if prev_pos == 0:
                with torch.no_grad():
                    out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos])
                    if other_model is not None:
                        other_model.model.set_compare(True)
                        out_compare = model.forward(tokens[:, cur_pos:cur_pos+1], past_key_values=out.past_key_values, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos+1])
                        other_model.model.set_compare(False)
                    if other_model is not None:
                        for i in range(len(other_model.lora_list)):
                            other_model.model.get_submodule(other_model.lora_list[i]).initialize_lora_weights()
            else:
                if other_model is not None:
                    loss = 0
                    optimizer = torch.optim.Adam(other_model.decode_optim_parameters(), 0.1)

                    if need_train == 0:
                        other_model.model.set_decode_training(True)
                        for e_g in range(0):
                            out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos])
                            loss = other_model.model.get_decode_loss()
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        other_model.model.set_decode_training(False)

                with torch.no_grad():
                    if other_model is not None:
                        if one:
                            other_model.model.set_last_input_token(True)
                        out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos])
                        if one:
                            other_model.model.set_last_input_token(False)

                        other_model.model.cal_prob(True)

                        if one and not other_model.model.get_search():

                            for a_tok_list in range(len(success_answers)):
                                tokens_clone = tokens.clone()
                                out_clone = copy.deepcopy(out)
                                prev_pos_clone = copy.deepcopy(prev_pos)
                                cur_pos_clone = copy.deepcopy(cur_pos)
                                a_prob = 0
                                if len(success_answers[a_tok_list]) + cur_pos_clone - 1 < tokens_clone.shape[1]:
                                    other_model.model.copy_matrix()
                                    a_prob = 0
                                    for a_tok in range(len(success_answers[a_tok_list])):
                                        a_prob = a_prob + F.softmax(out_clone.logits, dim=2)[0, -1, success_answers[a_tok_list][a_tok]]
                                        next_token_clone = torch.where(
                                            input_text_mask[0, cur_pos_clone], tokens_clone[0, cur_pos_clone], success_answers[a_tok_list][a_tok]
                                        )
                                        tokens_clone[:, cur_pos_clone] = next_token_clone
                                        prev_pos_clone = cur_pos_clone
                                        cur_pos_clone = cur_pos_clone + 1
                                        kvcache_clone = out_clone.past_key_values
                                        out_clone = model.forward(tokens_clone[:, prev_pos_clone:cur_pos_clone], past_key_values=kvcache_clone, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos_clone])
                                    a_prob = a_prob / len(success_answers[a_tok_list])
                                    other_model.model.reset_matrix()
                                prob_list_list[a_tok_list].append(a_prob)

                            for a_tok_list in range(len(false_answers)):
                                tokens_clone = tokens.clone()
                                out_clone = copy.deepcopy(out)
                                prev_pos_clone = copy.deepcopy(prev_pos)
                                cur_pos_clone = copy.deepcopy(cur_pos)
                                a_prob = 0
                                if len(false_answers[a_tok_list]) + cur_pos_clone - 1 < tokens_clone.shape[1]:
                                    other_model.model.copy_matrix()
                                    a_prob = 0
                                    for a_tok in range(len(false_answers[a_tok_list])):
                                        a_prob = a_prob + F.softmax(out_clone.logits, dim=2)[0, -1, false_answers[a_tok_list][a_tok]]
                                        next_token_clone = torch.where(
                                            input_text_mask[0, cur_pos_clone], tokens_clone[0, cur_pos_clone], false_answers[a_tok_list][a_tok]
                                        )
                                        tokens_clone[:, cur_pos_clone] = next_token_clone
                                        prev_pos_clone = cur_pos_clone
                                        cur_pos_clone = cur_pos_clone + 1
                                        kvcache_clone = out_clone.past_key_values
                                        out_clone = model.forward(tokens_clone[:, prev_pos_clone:cur_pos_clone], past_key_values=kvcache_clone, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos_clone])
                                    a_prob = a_prob / len(false_answers[a_tok_list])
                                    other_model.model.reset_matrix()
                                prob_list_list_false[a_tok_list].append(a_prob)
                        one = False

                        other_model.model.cal_prob(False)

            kvcache = out.past_key_values
            cache_list.append(out.attentions)

            if cur_pos > min_prompt_len:
                comma_token_id = mt.tokenizer.encode(",", add_special_tokens=False)[0]
                comma_detected = False
                for g_t in range(min_prompt_len, cur_pos):
                    if tokens[0][g_t] == comma_token_id:
                        comma_detected = True
                        break

                if comma_detected:
                    for g_t in range(min_prompt_len, cur_pos):
                        out.logits[:, -1, tokens[0][g_t]] = out.logits[:, -1, tokens[0][g_t]] / PEN

            next_token = torch.argmax(out.logits[:, -1], dim=-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            for stop_token in [mt.tokenizer.eos_token_id]:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)

        if len(prob_list_list) == 0:
            success_prob = 1
        else:
            success_prob = sum([max(i) for i in prob_list_list]) / len(prob_list_list)
        if len(prob_list_list_false) == 0:
            false_prob = 1
        else:
            false_prob = sum([max(i) for i in prob_list_list_false]) / len(prob_list_list_false)
        return tokens, cache_list, success_prob, false_prob


def debias_states_list(editor, model, num_layers, inp, test_inp, test_inp_rephrase, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list, prompt, prompt_test, prompt_test_rephrase, subject, relation, target_new, target_new_tok, subject_tok, relation_tok, answer_tok, train):
    global LABEL_TOK
    success_prob, false_prob = 0, 0

    states_to_patch = []
    for tnum in test_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(model, layer)))

    assert isinstance(target_new, list)
    answer_list_all = list()
    answer_toks_list_all = list()
    for edit_index in range(len(target_new)):
        if edit_index >= 2:
            break
        answer_list_all.append(target_new[edit_index])
        answer_toks_list_all.append(target_new_tok[edit_index])
    answer_list_all.extend(answer_list)
    answer_toks_list_all.extend(answer_toks_list)

    assert isinstance(target_new, list)
    target_new_temp = list()
    target_new_tok_temp = list()
    for edit_index in range(len(target_new)):
        if edit_index >= 2:
            break
        if train:
            edited_model, label_tok = editor.simple_edit(
                prompts=[prompt],
                target_new=[target_new[edit_index]],
                subject_tok=subject_tok,
                relation_tok=relation_tok,
                answer_tok=answer_tok,
                subject=[subject],
                mem_requests=memory_prompts,
                success_answers=answer_list
                )

            LABEL_TOK.append(label_tok.detach().cpu().numpy())
            editor.model = edited_model
        else:
            edited_model = editor.model
        target_new_temp.append(target_new[edit_index])
        target_new_tok_temp.append(target_new_tok[edit_index])

    states_to_patch = []
    for tnum in test_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(edited_model.model.model, layer)))
    if not train:
        outputs_exp_post, outputs_rephrase_post, answer_toks_list_all, answer_list_all, outputs_for_save_post, outputs_for_resubmit_post, outputs_for_resubmit_resubmit_post, outputs_for_resubmit_gen_post, cache_for_attentionmap_post, success_prob, false_prob = trace_with_patch_list(
            edited_model.model.model,
            test_inp,
            test_inp_rephrase,
            states_to_patch,
            answer_toks_list_all,
            answer_list_all,
            prompt_test,
            prompt_test_rephrase,
            state="post",
            addi=target_new[0:2],
            addi_tok=target_new_tok[0:2],
            other_model=edited_model,
            success_answers=answer_toks_list,
            false_answers=answer_toks_list_all[0:2],
            inp_for_search=inp
        )

    if not train:
        gen_text_post = [mt.tokenizer.decode(x, skip_special_tokens=False) for x in outputs_exp_post.detach().cpu().numpy().tolist()[0]]
        gen_text_1_post = ''
        char_tok_dict_post = dict()
        j = 0
        for t in range(len(gen_text_post)):
            t_1 = gen_text_post[t]
            for i in range(len(t_1)):
                gen_text_1_post = gen_text_1_post + t_1[i]
                char_tok_dict_post[j] = t
                j = j + 1

    if not train:
        gen_text_1_save = ""
        gen_text_1_resubmit = ""
        gen_text_1_resubmit_resubmit = ""
        gen_text_1_resubmit_gen = ""
        char_tok_dict_resubmit = dict()

        multi_results["pre"]["generate"].append(
            {
                "prompt": prompt_test,
                "gen_text_save": gen_text_1_save,
                "gen_text_resubmit": gen_text_1_resubmit,
                "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit,
                "gen_text_resubmit_gen": gen_text_1_resubmit_gen,
                "answer_list_all": answer_list_all,
                "answer_list_success": answer_list,
                "answer_list_miss": target_new_temp,
                "char_tok_dict_resubmit": char_tok_dict_resubmit,
                "insert_tok": str(test_inp["input_ids"].shape[1]-1),
            }
        )

        gen_text_1_rephrase_post = ""
        gen_text_1_save_post = ""
        gen_text_1_resubmit_post = ""
        gen_text_1_resubmit_resubmit_post = ""
        gen_text_1_resubmit_gen_post = ""
        char_tok_dict_resubmit_post = dict()

        multi_results["post"]["generate"].append(
            {
                "prompt": prompt_test,
                "gen_tok": outputs_exp_post.detach().cpu().numpy().tolist()[0],
                "gen_text": gen_text_1_post,
                "gen_text_rephrase": gen_text_1_rephrase_post,
                "gen_text_save": gen_text_1_save_post,
                "gen_text_resubmit": gen_text_1_resubmit_post,
                "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit_post,
                "gen_text_resubmit_gen": gen_text_1_resubmit_gen_post,
                "answer_list_all": answer_list_all,
                "answer_list_success": answer_list,
                "answer_list_miss": target_new_temp,
                "char_tok_dict": char_tok_dict_post,
                "char_tok_dict_resubmit": char_tok_dict_resubmit_post,
                "insert_tok": str(test_inp["input_ids"].shape[1]-1),
                "overlap": edited_model.model.base_model.get_overlap(),
                "contra": edited_model.model.base_model.get_contra()
            }
        )

    return success_prob, false_prob

def debias_states_list_port(editor, model, num_layers, hop1_inp, hop2_inp, reasoning_inp, reasoning_inp_search, hop1_check_tok_ids, hop2_check_tok_ids, reasoning_check_tok_ids, hop1_prompt, hop2_prompt, reasoning_prompt, implicit_answer, explicit_answer, explicit_answer_2, implicit_toks, explicit_toks, explicit_toks_2, hop1_subject_tok, hop1_relation_tok, hop1_answer_tok, hop2_subject_tok, hop2_relation_tok, hop2_answer_tok, origin_answer_list, train):
    global LABEL_TOK
    no_1 = False
    no_2 = False
    success_prob, false_prob = 0, 0

    states_to_patch = []
    for tnum in reasoning_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(model, layer)))

    if train:
        if not no_1:
            edited_model, label_tok = editor.simple_edit(
                prompts=[hop1_prompt],
                target_new=implicit_answer,
                subject_tok=hop1_subject_tok,
                relation_tok=hop1_relation_tok,
                answer_tok=hop1_answer_tok,
                subject=[""],
                mem_requests=memory_prompts,
                success_answers=list()
                )
            LABEL_TOK.append(label_tok.detach().cpu().numpy())
            editor.model = edited_model

        if not no_2:
            edited_model, label_tok = editor.simple_edit(
                prompts=[hop2_prompt],
                target_new=explicit_answer,
                subject_tok=hop2_subject_tok,
                relation_tok=hop2_relation_tok,
                answer_tok=hop2_answer_tok,
                subject=[""],
                mem_requests=memory_prompts,
                success_answers=list()
                )
            LABEL_TOK.append(label_tok.detach().cpu().numpy())
            editor.model = edited_model
    else:
        edited_model = editor.model

    states_to_patch = []
    for tnum in reasoning_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(edited_model.model.model, layer)))
    if not train:
        if no_1:
            success_answers = origin_answer_list[0]
            false_answers = explicit_answer
        elif no_2:
            success_answers = origin_answer_list[1]
            false_answers = implicit_answer
        else:
            success_answers = list()
            false_answers = [implicit_answer[0], explicit_answer[0]]
        outputs_exp_post, outputs_rephrase_post, explicit_toks_nouse, explicit_answer_nouse, outputs_for_save_post, outputs_for_resubmit_post, outputs_for_resubmit_resubmit_post, outputs_for_resubmit_gen_post, cache_for_attentionmap_post, success_prob, false_prob  = trace_with_patch_list(
            edited_model.model.model,
            reasoning_inp,
            reasoning_inp,
            states_to_patch,
            explicit_toks_2,
            explicit_answer_2,
            reasoning_prompt,
            reasoning_prompt,
            state="post",
            addi=list(),
            addi_tok=list(),
            other_model=edited_model,
            success_answers=success_answers,
            false_answers=false_answers,
            inp_for_search=reasoning_inp_search
        )
    if not train:
        begin_decode_tok = edited_model.model.base_model.get_begin_decode_tok()
        outputs_exp_post = outputs_exp_post[:, :]

        gen_text_post = [mt.tokenizer.decode(x, skip_special_tokens=False) for x in outputs_exp_post.detach().cpu().numpy().tolist()[0]]
        gen_text_1_post = ''
        char_tok_dict_post = dict()
        j = 0
        for t in range(len(gen_text_post)):
            t_1 = gen_text_post[t]
            for i in range(len(t_1)):
                gen_text_1_post = gen_text_1_post + t_1[i]
                char_tok_dict_post[j] = t
                j = j + 1

    if not train:
        gen_text_1_save = ""
        gen_text_1_resubmit = ""
        gen_text_1_resubmit_resubmit = ""
        gen_text_1_resubmit_gen = ""
        char_tok_dict_resubmit = dict()

        multi_results["pre"]["generate"].append(
            {
                "prompt": reasoning_prompt,
                "gen_text_save": gen_text_1_save,
                "gen_text_resubmit": gen_text_1_resubmit,
                "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit,
                "gen_text_resubmit_gen": gen_text_1_resubmit_gen,
                "answer_list_miss": [implicit_answer[0], explicit_answer[0]],
                "char_tok_dict_resubmit": char_tok_dict_resubmit,
                "insert_tok": str(reasoning_inp["input_ids"].shape[1]-1)
            }
        )

        gen_text_1_rephrase_post = ""
        gen_text_1_save_post = ""
        gen_text_1_resubmit_post = ""
        gen_text_1_resubmit_resubmit_post = ""
        gen_text_1_resubmit_gen_post = ""
        char_tok_dict_resubmit_post = dict()

        multi_results["post"]["generate"].append(
            {
                "prompt": reasoning_prompt,
                "gen_tok": outputs_exp_post.detach().cpu().numpy().tolist()[0],
                "gen_text": gen_text_1_post,
                "gen_text_save": gen_text_1_save_post,
                "gen_text_resubmit": gen_text_1_resubmit_post,
                "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit_post,
                "gen_text_resubmit_gen": gen_text_1_resubmit_gen_post,
                "answer_list_miss": [implicit_answer[0], explicit_answer[0]],
                "char_tok_dict": char_tok_dict_post,
                "char_tok_dict_resubmit": char_tok_dict_resubmit_post,
                "insert_tok": str(reasoning_inp["input_ids"].shape[1]-1)
            }
        )

    return success_prob, false_prob


def calculate_hidden_generation_flow(
    editor, mt, prompt, prompt_test, prompt_test_rephrase, subject, relation, target_new, target_new_tok, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list, train=False
    ):
    """
    Runs the A3E editing and evaluation pipeline for multi-answer editing.
    """
    inp = make_inputs(mt.tokenizer, [prompt])
    test_inp = make_inputs(mt.tokenizer, [prompt_test])
    test_inp_rephrase = make_inputs(mt.tokenizer, [prompt_test_rephrase])

    token_list = mt.tokenizer.encode(prompt)
    gen_text = [mt.tokenizer.decode(x, skip_special_tokens=False) for x in token_list]
    gen_text_1 = ''
    char_tok_dict = dict()
    j = 0
    for t in range(len(gen_text)):
        t_1 = gen_text[t]
        for i in range(len(t_1)):
            gen_text_1 = gen_text_1 + t_1[i]
            char_tok_dict[j] = t
            j = j + 1

    subject_position, subject_position_before = find_sublist_position_inprompt(gen_text_1, subject)
    subject_tok, subject_tok_before = char_tok_dict[subject_position], char_tok_dict[subject_position_before]

    relation_position, relation_position_before = find_sublist_position_inprompt(gen_text_1, relation)
    relation_tok, relation_tok_before = char_tok_dict[relation_position], char_tok_dict[relation_position_before]

    success_prob, false_prob = debias_states_list(editor, mt.model, mt.num_layers, inp, test_inp, test_inp_rephrase, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list, prompt, prompt_test, prompt_test_rephrase, subject, relation, target_new, target_new_tok, subject_tok, relation_tok, list(), train=train)

    return success_prob, false_prob


def calculate_hidden_generation_flow_port(
    editor, mt, hop1_prompt, hop2_prompt, hop1_subject, hop2_subject, hop1_relation, hop2_relation, reasoning_prompt, reasoning_prompt_search, hop1_check_tok_ids, hop2_check_tok_ids, reasoning_check_tok_ids, implicit_answer, explicit_answer, explicit_answer_2, implicit_toks, explicit_toks, explicit_toks_2, origin_answer_list, train=False
    ):
    """
    Runs the A3E editing and evaluation pipeline for portability/sequential editing.
    """
    hop1_inp = make_inputs(mt.tokenizer, [hop1_prompt])
    hop2_inp = make_inputs(mt.tokenizer, [hop2_prompt])
    reasoning_inp = make_inputs(mt.tokenizer, [reasoning_prompt])
    reasoning_inp_search = make_inputs(mt.tokenizer, [reasoning_prompt_search])

    hop1_token_list = mt.tokenizer.encode(hop1_prompt)
    hop2_token_list = mt.tokenizer.encode(hop2_prompt)
    hop1_gen_text = [mt.tokenizer.decode(x, skip_special_tokens=False) for x in hop1_token_list]
    hop2_gen_text = [mt.tokenizer.decode(x, skip_special_tokens=False) for x in hop2_token_list]

    hop1_gen_text_1 = ''
    hop1_char_tok_dict = dict()
    j = 0
    for t in range(len(hop1_gen_text)):
        t_1 = hop1_gen_text[t]
        for i in range(len(t_1)):
            hop1_gen_text_1 = hop1_gen_text_1 + t_1[i]
            hop1_char_tok_dict[j] = t
            j = j + 1

    hop2_gen_text_1 = ''
    hop2_char_tok_dict = dict()
    j = 0
    for t in range(len(hop2_gen_text)):
        t_1 = hop2_gen_text[t]
        for i in range(len(t_1)):
            hop2_gen_text_1 = hop2_gen_text_1 + t_1[i]
            hop2_char_tok_dict[j] = t
            j = j + 1

    hop1_subject_position, hop1_subject_position_before = 0,1
    hop1_subject_tok, hop1_subject_tok_before = hop1_char_tok_dict[hop1_subject_position], hop1_char_tok_dict[hop1_subject_position_before]

    hop1_relation_position, hop1_relation_position_before = 0,1
    hop1_relation_tok, hop1_relation_tok_before = hop1_char_tok_dict[hop1_relation_position], hop1_char_tok_dict[hop1_relation_position_before]

    hop2_subject_position, hop2_subject_position_before = 0,1
    hop2_subject_tok, hop2_subject_tok_before = hop2_char_tok_dict[hop2_subject_position], hop2_char_tok_dict[hop2_subject_position_before]

    hop2_relation_position, hop2_relation_position_before = 0,1
    hop2_relation_tok, hop2_relation_tok_before = hop2_char_tok_dict[hop2_relation_position], hop2_char_tok_dict[hop2_relation_position_before]

    success_prob, false_prob = debias_states_list_port(editor, mt.model, mt.num_layers, hop1_inp, hop2_inp, reasoning_inp, reasoning_inp_search, hop1_check_tok_ids, hop2_check_tok_ids, reasoning_check_tok_ids, hop1_prompt, hop2_prompt, reasoning_prompt, implicit_answer, explicit_answer, explicit_answer_2, implicit_toks, explicit_toks, explicit_toks_2, hop1_subject_tok, hop1_relation_tok, list(), hop2_subject_tok, hop2_relation_tok, list(), origin_answer_list, train=train)
    return success_prob, false_prob


# ============================================================
# Main: Initialize model, load data, run A3E editing pipeline
# ============================================================

hparams = A3EHyperParams.from_hparams('/home/hmpiao/EasyEdit_method/hparams/A3E/llama3-8b.yaml')
editor = BaseEditor.from_hparams(hparams)

mt = ModelAndTokenizer(
    tokenizer=editor.tok,
    model=editor.model,
    torch_dtype = editor.torch_dtype,
)

vocab_size = int(mt.model.state_dict()['lm_head.weight'].shape[0])

write_path = "/home/hmpiao/hmpiao/log/melo/gracenipsrebuttal_sequential_multi_analysis_all_continue_50edits_again.json"
data_path = "/home/hmpiao/EasyEdit_method/data/counterfact-edit.json"
memory_data_path = "/home/hmpiao/EasyEdit_method/data/benchmark_ZsRE_ZsRE-test-all.json"

fr = open(data_path, "r")
data = json.load(fr)

if "counterfact" in data_path:
    valid_index = []
    for i, edit_data_ in enumerate(data):
        if not edit_data_['target_new'] == '' and edit_data_['target_new'] is not None:
            valid_index.append(i)
    edit_data = [edit_data_ for i, edit_data_ in enumerate(data) if i in valid_index]
    print(f"len(edit_data): {len(edit_data)}")
    data = edit_data[:886]

memory_fr = open(memory_data_path, "r")
memory_data = json.load(memory_fr)
memory_data = random.sample(memory_data, 1000)
memory_prompts = list()
for d in memory_data:
    if "locality" in d.keys():
        for k in d["locality"].keys():
            for p in d["locality"][k]:
                memory_prompts.append(p["prompt"])

for key in mt.model.state_dict().keys():
    print(key)
W = copy.deepcopy(mt.model.state_dict()['lm_head.weight'])
Norm_W = copy.deepcopy(mt.model.state_dict()['model.norm.weight'])
Norm = copy.deepcopy(mt.model.model.norm)

output_data = list()

i = 0
edit_needed = 0
data_len = len(data)
success_prob_all, false_prob_all = list(), list()

# ============================================================
# Phase 1: Training - Apply A3E edits to the model
# ============================================================
while i < len(data):
    if i >= 885:
        i = i + 1
        continue

    if "port" in data_path:
        if i >= len(data) - 6:
            break
    elif "sequential" in data_path:
        if data[i]["id"] >= 296:
            break

    output_datum = dict()

    if "port" in data_path:
        assert data[i]["type"] == "hop1" \
            and data[i+1]["type"] == "hop2" \
            and data[i+2]["type"] == "reasoning"
        hop1_prompt = data[i]["text"]
        hop2_prompt = data[i+1]["text"]
        reasoning_prompt = data[i+2]["text"]
        reasoning_prompt_search = data[i+2]["text"]
        hop1_subject = data[i]["concept"]
        hop2_subject = data[i+1]["concept"]
        hop1_relation = data[i]["concept"]
        hop2_relation = data[i+1]["concept"]
        hop1_origin_answer = data[i]["origin_labels"]
        hop1_origin_answer = hop1_origin_answer.replace("(", "").replace(")", "")
        hop2_origin_answer = data[i+1]["origin_labels"]
        hop2_origin_answer = hop2_origin_answer.replace("(", "").replace(")", "")
        implicit_answer = data[i]["labels"]
        implicit_answer = implicit_answer.replace("(", "").replace(")", "")
        explicit_answer = data[i+2]["labels"]
        explicit_answer = explicit_answer.replace("(", "").replace(")", "")
        i = i + 3

        hop1_last_tok_id = get_tgt_tok_id(hop1_prompt)
        hop1_check_tok_ids = [hop1_last_tok_id]
        hop2_last_tok_id = get_tgt_tok_id(hop2_prompt)
        hop2_check_tok_ids = [hop2_last_tok_id]
        reasoning_last_tok_id = get_tgt_tok_id(reasoning_prompt)
        reasoning_check_tok_ids = [reasoning_last_tok_id]
        implicit_toks = enc_tok(implicit_answer, avg=True)
        explicit_toks = enc_tok(explicit_answer, avg=True)

        success_prob, false_prob = calculate_hidden_generation_flow_port(editor, mt, hop1_prompt, hop2_prompt, hop1_subject, hop2_subject, hop1_relation, hop2_relation, reasoning_prompt, reasoning_prompt_search, hop1_check_tok_ids, hop2_check_tok_ids\
                                              , reasoning_check_tok_ids, [implicit_answer], [explicit_answer], [explicit_answer], [implicit_toks], [explicit_toks], [explicit_toks], [hop1_origin_answer,hop2_origin_answer], train=True)

        mt.model = editor.model

    elif "multi" in data_path:
        multi_prompt = data[i]["text"]
        multi_prompt_test = data[i]["text"]
        multi_prompt_rephrase = data[i]["text"]
        multi_prompt_test_rephrase = multi_template_rephrase.format(multi_prompt_rephrase[0:-1])
        multi_subject = data[i]["concept"]
        multi_relation = data[i]["concept"]
        answer_list = []
        answer_list_new = data[i]["answer_miss"]
        for answer_id in range(len(answer_list)):
            answer_list[answer_id] = answer_list[answer_id].replace("(", "").replace(")", "").replace(".", "").replace(",", "").replace(" ", "")
        for answer_id_new in range(len(answer_list_new)):
            answer_list_new[answer_id_new] = answer_list_new[answer_id_new].replace("(", "").replace(")", "").replace(".", "").replace(",", "").replace(" ", "")

        i = i + 1

        last_tok_id = get_tgt_tok_id(multi_prompt)
        check_tok_ids = [last_tok_id]
        test_last_tok_id = get_tgt_tok_id(multi_prompt_test)
        test_check_tok_ids = [test_last_tok_id]

        answer_toks_list = list()
        for answer in answer_list:
            toks = enc_tok(answer, avg=True)
            answer_toks_list.append(toks)

        answer_toks_list_new = list()
        for answer_new in answer_list_new:
            toks_new = enc_tok(answer_new, avg=True)
            answer_toks_list_new.append(toks_new)

        success_prob, false_prob = calculate_hidden_generation_flow(editor, mt, multi_prompt, multi_prompt_test, multi_prompt_test_rephrase, multi_subject, multi_relation, answer_list_new, answer_toks_list_new, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list, train=True)

        mt.model = editor.model

    elif "dc" in data_path:
        seq1_prompt = data[i]["text"]
        seq2_prompt = data[i+1]["text"]
        sequential_prompt = sequential_template.format(seq1_prompt, seq2_prompt)
        sequential_prompt_search = sequential_template_search.format(seq1_prompt, seq2_prompt)
        seq1_prompt = sequential_template_edit.format(seq1_prompt)
        seq2_prompt = sequential_template_edit.format(seq2_prompt)
        seq1_subject = data[i]["concept"].replace(" ", "")
        seq2_subject = data[i+1]["concept"].replace(" ", "")
        seq1_relation = data[i]["concept"].replace(" ", "")
        seq2_relation = data[i+1]["concept"].replace(" ", "")
        seq1_answer = data[i]["answer_miss"][0].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        seq1_origin_answer = ""
        seq2_answer = data[i+1]["answer_miss"][0].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        seq2_origin_answer = ""

        sequential_answer = seq1_answer + ", " + seq2_answer

        i = i + 2

        seq1_last_tok_id = get_tgt_tok_id(seq1_prompt)
        seq1_check_tok_ids = [seq1_last_tok_id]
        seq2_last_tok_id = get_tgt_tok_id(seq2_prompt)
        seq2_check_tok_ids = [seq2_last_tok_id]
        sequential_last_tok_id = get_tgt_tok_id(sequential_prompt)
        sequential_check_tok_ids = [sequential_last_tok_id]
        seq1_toks = enc_tok(seq1_answer, avg=True)
        seq2_toks = enc_tok(seq2_answer, avg=True)
        sequential_toks = enc_tok(sequential_answer, avg=True)

        success_prob, false_prob = calculate_hidden_generation_flow_port(editor, mt, seq1_prompt, seq2_prompt, seq1_subject, seq2_subject, seq1_relation, seq2_relation, sequential_prompt, sequential_prompt_search, seq1_check_tok_ids, seq2_check_tok_ids\
                                              , sequential_check_tok_ids, [seq1_answer], [seq2_answer], [sequential_answer], [seq1_toks], [seq2_toks], [sequential_toks], [seq1_origin_answer, seq2_origin_answer], train=True)

        mt.model = editor.model

    elif "counterfact" in data_path:
        seq1_prompt = data[i]["prompt"]
        seq2_prompt = data[i+1]["prompt"]
        sequential_prompt = sequential_template.format(seq1_prompt, seq2_prompt)
        sequential_prompt_search = sequential_template_search.format(seq1_prompt, seq2_prompt)
        seq1_prompt = sequential_template_edit.format(seq1_prompt)
        seq2_prompt = sequential_template_edit.format(seq2_prompt)
        seq1_subject = data[i]["subject"].replace(" ", "")
        seq2_subject = data[i+1]["subject"].replace(" ", "")
        seq1_relation = data[i]["subject"].replace(" ", "")
        seq2_relation = data[i+1]["subject"].replace(" ", "")
        seq1_answer = data[i]["target_new"].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        seq1_origin_answer = ""
        seq2_answer = data[i+1]["target_new"].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        seq2_origin_answer = ""

        sequential_answer = seq1_answer + ", " + seq2_answer

        i = i + 2

        seq1_last_tok_id = get_tgt_tok_id(seq1_prompt)
        seq1_check_tok_ids = [seq1_last_tok_id]
        seq2_last_tok_id = get_tgt_tok_id(seq2_prompt)
        seq2_check_tok_ids = [seq2_last_tok_id]
        sequential_last_tok_id = get_tgt_tok_id(sequential_prompt)
        sequential_check_tok_ids = [sequential_last_tok_id]
        seq1_toks = enc_tok(seq1_answer, avg=True)
        seq2_toks = enc_tok(seq2_answer, avg=True)
        sequential_toks = enc_tok(sequential_answer, avg=True)

        success_prob, false_prob = calculate_hidden_generation_flow_port(editor, mt, seq1_prompt, seq2_prompt, seq1_subject, seq2_subject, seq1_relation, seq2_relation, sequential_prompt, sequential_prompt_search, seq1_check_tok_ids, seq2_check_tok_ids\
                                              , sequential_check_tok_ids, [seq1_answer], [seq2_answer], [sequential_answer], [seq1_toks], [seq2_toks], [sequential_toks], [seq1_origin_answer, seq2_origin_answer], train=True)

        mt.model = editor.model


# ============================================================
# Save/Load checkpoint
# ============================================================
globalparameter_dict = editor.model.model.base_model.get_global()
torch.save(globalparameter_dict, '/home/hmpiao/hmpiao/chkt/multi_grace_edited_global_50edits_rebuttal.pth')
torch.save(mt.model.state_dict(), '/home/hmpiao/hmpiao/chkt/multi_grace_edited_50edits_rebuttal.pth')
globalparameter_dict = torch.load('/home/hmpiao/hmpiao/chkt/multi_grace_edited_global_50edits_rebuttal.pth')
editor.model.model.base_model.set_global(globalparameter_dict)
mt.model.load_state_dict(torch.load('/home/hmpiao/hmpiao/chkt/multi_grace_edited_50edits_rebuttal.pth', map_location=torch.device("cpu")))
editor.model = mt.model

if "local" in write_path:
    data_path = "/home/hmpiao/EasyEdit_method/data/demo_dc_analysis_all_rephrase_local_localanswer.json"
    fr = open(data_path, "r")
    data = json.load(fr)

# ============================================================
# Phase 2: Evaluation - Test the edited model
# ============================================================
temporature = [0.02]
penalty = [10.0]
for tem in temporature:
    for pen in penalty:
        ONLY_ONE_QUESTION = list()
        I = -1
        multi_results = dict()
        multi_results["pre"] = dict()
        multi_results["post"] = dict()
        multi_results["pre"]["origin"] = dict()
        multi_results["pre"]["hopping"] = dict()
        multi_results["post"]["origin"] = dict()
        multi_results["post"]["hopping"] = dict()
        multi_results["pre"]["origin"]["prob"] = dict()
        multi_results["pre"]["origin"]["rank"] = dict()
        multi_results["pre"]["hopping"]["prob"] = dict()
        multi_results["pre"]["hopping"]["rank"] = dict()
        multi_results["post"]["origin"]["prob"] = dict()
        multi_results["post"]["origin"]["rank"] = dict()
        multi_results["post"]["hopping"]["prob"] = dict()
        multi_results["post"]["hopping"]["rank"] = dict()
        multi_results["pre"]["generate"] = list()
        multi_results["post"]["generate"] = list()
        TEM = tem
        PEN = pen
        write_path_copy = write_path + "_" + str(tem) + "_" + str(pen) + ".json"

        i = 0
        while i < len(data):
            if i >= 885:
                i = i + 1
                continue

            if "port" in data_path:
                if i >= len(data) - 6:
                    break
            elif "sequential" in data_path:
                if data[i]["id"] >= 296:
                    break

            output_datum = dict()

            if "port" in data_path:
                assert data[i]["type"] == "hop1" \
                    and data[i+1]["type"] == "hop2" \
                    and data[i+2]["type"] == "reasoning"
                hop1_prompt = data[i]["text"]
                hop2_prompt = data[i+1]["text"]
                reasoning_prompt = data[i+2]["text"]
                reasoning_prompt_search = data[i+2]["text"]
                hop1_subject = data[i]["concept"]
                hop2_subject = data[i+1]["concept"]
                hop1_relation = data[i]["relation"]
                hop2_relation = data[i+1]["relation"]
                hop1_origin_answer = data[i]["origin_labels"].replace("(", "").replace(")", "")
                hop2_origin_answer = data[i+1]["origin_labels"].replace("(", "").replace(")", "")
                implicit_answer = data[i]["labels"].replace("(", "").replace(")", "")
                explicit_answer = data[i+2]["labels"].replace("(", "").replace(")", "")
                I = I + 1
                i = i + 3

                hop1_last_tok_id = get_tgt_tok_id(hop1_prompt)
                hop1_check_tok_ids = [hop1_last_tok_id]
                hop2_last_tok_id = get_tgt_tok_id(hop2_prompt)
                hop2_check_tok_ids = [hop2_last_tok_id]
                reasoning_last_tok_id = get_tgt_tok_id(reasoning_prompt)
                reasoning_check_tok_ids = [reasoning_last_tok_id]
                implicit_toks = enc_tok(implicit_answer, avg=True)
                explicit_toks = enc_tok(explicit_answer, avg=True)

                success_prob, false_prob = calculate_hidden_generation_flow_port(editor, mt, hop1_prompt, hop2_prompt, hop1_subject, hop2_subject, hop1_relation, hop2_relation, reasoning_prompt, reasoning_prompt_search, hop1_check_tok_ids, hop2_check_tok_ids\
                                                    , reasoning_check_tok_ids, [implicit_answer], [explicit_answer], [explicit_answer], [implicit_toks], [explicit_toks], [explicit_toks], [hop1_origin_answer,hop2_origin_answer])

                print(f"success answers prob for {i}: {success_prob}")
                print(f"false answers prob for {i}: {false_prob}")
                success_prob_all.append(success_prob)
                false_prob_all.append(false_prob)
                mt.model = editor.model
                json.dump(multi_results, open(write_path_copy, "w"), indent=4)

            elif "multi" in data_path or "local" in write_path:
                if "local" in write_path:
                    multi_prompt = data[i]["locality"][0][0]
                    multi_prompt_test = multi_prompt
                    multi_prompt_rephrase = data[i]["locality"][0][0]
                    multi_prompt_test_rephrase = multi_prompt_rephrase
                    multi_subject = " "
                    multi_relation = " "
                    answer_list = []
                    answer_list_new = [data[i]["locality_answer"], data[i]["locality_answer"]]
                else:
                    multi_prompt = data[i]["text"]
                    multi_prompt_test = multi_template.format(multi_prompt)
                    multi_prompt_rephrase = data[i]["text"]
                    multi_prompt_test_rephrase = multi_template_rephrase.format(multi_prompt_rephrase[0:-1])
                    multi_subject = data[i]["concept"]
                    multi_relation = data[i]["concept"]
                    answer_list = []
                    answer_list_new = data[i]["answer_miss"]
                for answer_id in range(len(answer_list)):
                    answer_list[answer_id] = answer_list[answer_id].replace("(", "").replace(")", "").replace(".", "").replace(",", "").replace(" ", "")
                for answer_id_new in range(len(answer_list_new)):
                    answer_list_new[answer_id_new] = answer_list_new[answer_id_new].replace("(", "").replace(")", "").replace(".", "").replace(",", "").replace(" ", "")
                if len(answer_list_new) < 2 and "local" not in write_path:
                    ONLY_ONE_QUESTION.append(i)
                    print(f"There is one only! {ONLY_ONE_QUESTION}")
                    I = I + 1
                    i = i + 1
                    continue
                I = I + 1
                i = i + 1

                last_tok_id = get_tgt_tok_id(multi_prompt)
                check_tok_ids = [last_tok_id]
                test_last_tok_id = get_tgt_tok_id(multi_prompt_test)
                test_check_tok_ids = [test_last_tok_id]

                answer_toks_list = list()
                for answer in answer_list:
                    toks = enc_tok(answer, avg=True)
                    answer_toks_list.append(toks)

                answer_toks_list_new = list()
                for answer_new in answer_list_new:
                    toks_new = enc_tok(answer_new, avg=True)
                    answer_toks_list_new.append(toks_new)

                success_prob, false_prob = calculate_hidden_generation_flow(editor, mt, multi_prompt, multi_prompt_test, multi_prompt_test_rephrase, multi_subject, multi_relation, answer_list_new, answer_toks_list_new, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list)
                print(f"success answers prob for {i}: {success_prob}")
                print(f"false answers prob for {i}: {false_prob}")
                if len(answer_list) != 0:
                    success_prob_all.append(success_prob)
                false_prob_all.append(false_prob)
                mt.model = editor.model
                json.dump(multi_results, open(write_path_copy, "w"), indent=4)

            elif "dc" in data_path:
                seq1_prompt = data[i]["text"]
                seq2_prompt = data[i+1]["text"]
                sequential_prompt = sequential_template.format(seq1_prompt, seq2_prompt)
                sequential_prompt_search = sequential_template_search.format(seq1_prompt, seq2_prompt)
                seq1_prompt = sequential_template_edit.format(seq1_prompt)
                seq2_prompt = sequential_template_edit.format(seq2_prompt)
                seq1_subject = data[i]["concept"].replace(" ", "")
                seq2_subject = data[i+1]["concept"].replace(" ", "")
                seq1_relation = data[i]["concept"].replace(" ", "")
                seq2_relation = data[i+1]["concept"].replace(" ", "")
                seq1_answer = data[i]["answer_miss"][0].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
                seq1_origin_answer = ""
                seq2_answer = data[i+1]["answer_miss"][0].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
                seq2_origin_answer = ""

                sequential_answer = seq1_answer + ", " + seq2_answer

                I = I + 1
                i = i + 2

                seq1_last_tok_id = get_tgt_tok_id(seq1_prompt)
                seq1_check_tok_ids = [seq1_last_tok_id]
                seq2_last_tok_id = get_tgt_tok_id(seq2_prompt)
                seq2_check_tok_ids = [seq2_last_tok_id]
                sequential_last_tok_id = get_tgt_tok_id(sequential_prompt)
                sequential_check_tok_ids = [sequential_last_tok_id]
                seq1_toks = enc_tok(seq1_answer, avg=True)
                seq2_toks = enc_tok(seq2_answer, avg=True)
                sequential_toks = enc_tok(sequential_answer, avg=True)

                success_prob, false_prob = calculate_hidden_generation_flow_port(editor, mt, seq1_prompt, seq2_prompt, seq1_subject, seq2_subject, seq1_relation, seq2_relation, sequential_prompt, sequential_prompt_search, seq1_check_tok_ids, seq2_check_tok_ids\
                                                    , sequential_check_tok_ids, [seq1_answer], [seq2_answer], [sequential_answer], [seq1_toks], [seq2_toks], [sequential_toks], [seq1_origin_answer, seq2_origin_answer])

                print(f"success answers prob for {i}: {success_prob}")
                print(f"false answers prob for {i}: {false_prob}")
                if False:
                    success_prob_all.append(success_prob)
                false_prob_all.append(false_prob)
                mt.model = editor.model
                json.dump(multi_results, open(write_path_copy, "w"), indent=4)

            elif "counterfact" in data_path:
                seq1_prompt = data[i]["rephrase_prompt"]
                seq2_prompt = data[i+1]["rephrase_prompt"]
                sequential_prompt = seq1_prompt
                sequential_prompt_search = seq1_prompt
                seq1_prompt = seq1_prompt
                seq2_prompt = seq1_prompt
                seq1_subject = data[i]["subject"].replace(" ", "")
                seq2_subject = data[i+1]["subject"].replace(" ", "")
                seq1_relation = data[i]["subject"].replace(" ", "")
                seq2_relation = data[i+1]["subject"].replace(" ", "")
                seq1_answer = data[i]["target_new"].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
                seq1_origin_answer = ""
                seq2_answer = data[i+1]["target_new"].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
                seq2_origin_answer = ""

                sequential_answer = seq1_answer

                I = I + 1
                i = i + 1

                seq1_last_tok_id = get_tgt_tok_id(seq1_prompt)
                seq1_check_tok_ids = [seq1_last_tok_id]
                seq2_last_tok_id = get_tgt_tok_id(seq2_prompt)
                seq2_check_tok_ids = [seq2_last_tok_id]
                sequential_last_tok_id = get_tgt_tok_id(sequential_prompt)
                sequential_check_tok_ids = [sequential_last_tok_id]
                seq1_toks = enc_tok(seq1_answer, avg=True)
                seq2_toks = enc_tok(seq2_answer, avg=True)
                sequential_toks = enc_tok(sequential_answer, avg=True)

                success_prob, false_prob = calculate_hidden_generation_flow_port(editor, mt, seq1_prompt, seq2_prompt, seq1_subject, seq2_subject, seq1_relation, seq2_relation, sequential_prompt, sequential_prompt_search, seq1_check_tok_ids, seq2_check_tok_ids\
                                                    , sequential_check_tok_ids, [seq1_answer], [seq2_answer], [sequential_answer], [seq1_toks], [seq2_toks], [sequential_toks], [seq1_origin_answer, seq2_origin_answer])

                print(f"success answers prob for {i}: {success_prob}")
                print(f"false answers prob for {i}: {false_prob}")
                if False:
                    success_prob_all.append(success_prob)
                false_prob_all.append(false_prob)
                mt.model = editor.model
                json.dump(multi_results, open(write_path_copy, "w"), indent=4)

        if len(success_prob_all) > 0:
            print(sum(success_prob_all)/len(success_prob_all))
        print(sum(false_prob_all)/len(false_prob_all))
        json.dump(multi_results, open(write_path_copy, "w"), indent=4)
