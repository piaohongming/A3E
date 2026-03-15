import os, re, json, sys
# os.chdir("/root/autodl-tmp/zhaoyi/knowledge_locate/rome")
sys.path.append("../..") 
import torch, numpy
from collections import defaultdict
from easyeditor.util import nethook
from easyeditor.editors import BaseEditor
from easyeditor.models import MELOHyperParams, GraceHyperParams

#sys.path.append("../..")
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
import matplotlib.pyplot as plt
import copy
import transformer_lens
import transformer_lens.utils as utils
import matplotlib.pyplot as plt

def imshow(tensor, layer, state, gen):
    for i in range(tensor.shape[0]):
        if tensor.shape[1] > 1:
            plt.matshow(utils.to_numpy(tensor[i][-14:,-14:]), cmap=plt.cm.Blues)
        else:
            plt.matshow(utils.to_numpy(tensor[i][:,-14:]), cmap=plt.cm.Blues)
        #plt.savefig("/home/hmpiao/rome/experiments/picture/grace_attention_{}_l{}_h{}_t{}.png".format(state, layer, i, gen))
        plt.savefig("/data2/hmpiao/FME/logits_len_loss_attention/tpatchertwo_attention_{}_l{}_h{}_t{}.png".format(state, layer, i, gen))
        plt.clf()
        plt.close()
    #fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)
    #pio.write_image(fig, './probe.jpg')


random.seed(0)
max_attention_dict = dict()
max_attention_dict_multi = dict()
max_attention_dict_post = dict()
for i in range(32):
    max_attention_dict[f"layer_{i}"] = dict()
    max_attention_dict_post[f"layer_{i}"] = dict()
    max_attention_dict[f"layer_{i}"]["context"] = 0
    max_attention_dict_post[f"layer_{i}"]["context"] = 0
    max_attention_dict[f"layer_{i}"]["subject"] = 0
    max_attention_dict_post[f"layer_{i}"]["subject"] = 0
    max_attention_dict[f"layer_{i}"]["relation"] = 0
    max_attention_dict_post[f"layer_{i}"]["relation"] = 0
    for p in range(3):
        max_attention_dict_multi[f"layer_{i}_position_{p}"] = dict()
        max_attention_dict_multi[f"layer_{i}_position_{p}"]["context"] = 0
        max_attention_dict_multi[f"layer_{i}_position_{p}"]["subject"] = 0
        max_attention_dict_multi[f"layer_{i}_position_{p}"]["relation"] = 0
        max_attention_dict_multi[f"layer_{i}_position_{p}"]["answer0"] = 0
        max_attention_dict_multi[f"layer_{i}_position_{p}"]["answer1"] = 0
        max_attention_dict_multi[f"layer_{i}_position_{p}"]["answer2"] = 0
    for j in range(32):
        max_attention_dict[f"layer_{i}_head_{j}"] = dict()
        max_attention_dict_post[f"layer_{i}_head_{j}"] = dict()
        max_attention_dict[f"layer_{i}_head_{j}"]["context"] = 0
        max_attention_dict_post[f"layer_{i}_head_{j}"]["context"] = 0
        max_attention_dict[f"layer_{i}_head_{j}"]["subject"] = 0
        max_attention_dict_post[f"layer_{i}_head_{j}"]["subject"] = 0
        max_attention_dict[f"layer_{i}_head_{j}"]["relation"] = 0
        max_attention_dict_post[f"layer_{i}_head_{j}"]["relation"] = 0
        for p in range(3):
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"] = dict()
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"]["context"] = 0
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"]["subject"] = 0
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"]["relation"] = 0
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"]["answer0"] = 0
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"]["answer1"] = 0
            max_attention_dict_multi[f"layer_{i}_head_{j}_position_{p}"]["answer2"] = 0
#torch.set_grad_enabled(False)

last_token_representation = None
last_token_representation_resubmit = None
last_token_representation_gen = None
x_before = None

t_global = None
t_global_change = None
t_global_hopping = None
t_global_change_hopping = None
port_type = ""
port_dict = {"explicit":{"prob":[], "rank":[]}, "implicit":{"prob":[], "rank":[]}}
port_results = dict()
port_results["hop2"] = copy.deepcopy(port_dict)
port_results["reasoning"] = copy.deepcopy(port_dict)

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
#multi_results["answer_dict"] = dict()

port_template = "Q: The name of the anthem of Japan is? A: Kimigayo\n\
    Q: The name of the head of government of the place of birth of Lolita Robertson is? A: London Breed\n\
        Q: The place of birth of Mikuru Suzuki is? A: Japan\n\
            Q: The name of the continent which the place of birth of Lolita Robertson is part of is? A: North America\n\
                Q: {}? A: "
multi_template = "Q: Tim Dorsey, who has written the? A: Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\n\
    Q: Jerusalem, which is the partner town of? A: NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\n\
        Q: Pushkin is the author of? A: The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\n\
            Q: WWE is the owner of? A: WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\n\
                Q: {}? A: "
multi_template_rephrase = "Q: Tim Dorsey, who has written the? A: Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\n\
    Q: Jerusalem, which is the partner town of? A: NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\n\
        Q: Pushkin is the author of? A: The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\n\
            Q: WWE is the owner of? A: WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\n\
                Q: {}? A: "
'''
sequential_template = "Q: The name of the head of government of San Francisco is and the name of the continent which San Francisco is? A: London Breed and North America\n\
    Q: The name of the continent which New York City is and the official language of the New York City is? A: North America and English\n\
        Q: The name of the head of government of New York City is and the name of the capital city of Japan is? A: Eric Adams and Tokyo\n\
            Q: The official language of Japan is and the name of the anthem of Japan is? A: Japanese and Kimigayo\n\
                Q: {}? A: "
'''

sequential_template = "Q1: The name of the anthem of Japan is? Q2: The capital of China is? A1: Kimigayo A2: Beijing\n\
            Q1: The continent of Japan is? Q2: The producer of iphone is? A1: Asia A2: Apple Inc\n\
                Q1: {}? Q2: {}? "

'''
sequential_template = "Q1: What is the name of the anthem of Japan? Q2: What is the capital of China? A1: Kimigayo A2: Beijing\n\
            Q1: What is the continent of Japan? Q2: What is the producer of iphone? A1: Asia A2: Apple Inc\n\
                Q1: {}? Q2: {}? "
'''
#sequential_template_search = "Q: The answers of the two questions ""{}"" and ""{}"" are? A: "
#sequential_template_search = "{} and {} is? "
'''
sequential_template_search = "Q: The answers of the two questions ""The name of the anthem of Japan is"" and ""The capital of China is"" are? A: Kimigayo and Beijing\n\
            Q: The answers of the two questions ""The continent of Japan is"" and ""The producer of iphone is"" are? A: Asia and Apple Inc\n\
                Q: The answers of the two questions ""{}"" and ""{}"" are? A: "
'''

sequential_template_search = "Q1: The name of the anthem of Japan is? Q2: The capital of China is? A1: Kimigayo A2: Beijing\n\
            Q1: The continent of Japan is? Q2: The producer of iphone is? A1: Asia A2: Apple Inc\n\
                Q1: {}? Q2: {}? "

'''
sequential_template_search = "Q1: What is the name of the anthem of Japan? Q2: What is the capital of China? A1: Kimigayo A2: Beijing\n\
            Q1: What is the continent of Japan? Q2: What is the producer of iphone? A1: Asia A2: Apple Inc\n\
                Q1: {}? Q2: {}? "
'''

def find_sublist_position(main_list, sub_list, prompt):
    position = 10000
    position_before = 10000
    '''
    if sub_list in prompt:
        return position, position_before
    '''
    for i, element in enumerate(main_list):
        # 检查当前元素是否为子列表的开始
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
        '''
        avg = True: return all of the encs of the answer string.
        '''
        print('check_tok_enc:', mt.tokenizer.encode(check_tok))
        # check_tok_enc = mt.tokenizer.encode(check_tok)[-1]
        if avg == False:
            check_tok_enc = mt.tokenizer.encode(check_tok)[1] # detach [SOS] token
        else:
            check_tok_enc = mt.tokenizer.encode(check_tok)[1:] # detach [SOS] token
        if isinstance(check_tok_enc, list):
            return check_tok_enc
        elif isinstance(check_tok_enc, int):
            return [check_tok_enc]
        else:
            print(check_tok_enc)
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
                            # check_rank.append(cnt)

                    temp_rank_sum = sum([v for v in temp_rank.values()])
                    return temp_rank_sum/len(check_tok_enc)

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    implicit_toks, # toks to debias
    explicit_toks, # toks to check 
    ):
    tgt_toks = implicit_toks

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    print(patch_spec.keys())


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x


    

    # Define the model-patching rule.
    def patch_rep(x, layer):
        global port_type
        global port_results
        '''
        tgt_toks = [tok_enc_1, tok_enc_2, ...]
        '''
        
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x) # h.ahspe = [bs, seq_len, hid_dim]
        #norm_x = Norm(x)
        #h = untuple(norm_x)
        print(layer)
        for t in patch_spec[layer]:
            # t is traced tok
            out = h[0, t].to(W.device)
            v = torch.matmul(W, out) # v.shape = [32000, 1] or [32000]?
            if len(v.shape) == 1:
                v = v.unsqueeze(1) # v.shape = [32000, 1]
            
            v = v.squeeze()
            logits = torch.softmax(v, dim=0).tolist()

            explicit_prob = 0
            for tok in explicit_toks:
                explicit_prob += logits[tok]
            explicit_prob /= len(explicit_toks)
            explicit_rank = get_rank(logits, explicit_toks)

            implicit_prob = 0
            for tok in implicit_toks:
                implicit_prob += logits[tok]
            implicit_prob /= len(implicit_toks)
            implicit_rank = get_rank(logits, implicit_toks)

            port_results[port_type]["explicit"]["prob"].append(explicit_prob)
            port_results[port_type]["explicit"]["rank"].append(explicit_rank)
            port_results[port_type]["implicit"]["prob"].append(implicit_prob)
            port_results[port_type]["implicit"]["rank"].append(implicit_rank)
            '''
            v_min = torch.min(v) # v_min.shape = []
            v_max = torch.max(v)
            delta_v = torch.zeros_like(v) # delta_v.shape = [32000, 1]
            delta_v_compare = torch.zeros_like(v)

            for k in tgt_toks:
                if mode == "suppress":
                    delta_v[k, 0] = v_min - v[k,0]
                elif mode == "enhance":
                    delta_v[k, 0] = max(v_max, 2*v[k,0]) - v[k,0]
                else:
                    raise Exception("Unexpected Mode:"+mode)

            comparsions = random.sample(range(0, vocab_size), len(tgt_toks))
            
            for k in comparsions:
                if mode == "suppress":
                    delta_v_compare[k, 0] = v_min - v[k,0]
                elif mode == "enhance":
                    delta_v_compare[k, 0] = max(v_max, 2*v[k,0]) - v[k,0]
                else:
                    raise Exception("Unexpected Mode:"+mode)

            delta_h = torch.matmul(A, delta_v.double()).squeeze(1) # delta_h.shape = [hid_dim]
            delta_h_compare = torch.matmul(A, delta_v_compare.double()).squeeze(1)

            h[0, t] += delta_h.half() # shape = [hid_dim]
            h[1, t] += delta_h_compare.half()

            # double check projection
            if recheck:
                v_update = torch.matmul(W, h[0, t]) # v_update.shape = [32000, 1] or [32000]
                if len(v_update.shape) == 1:
                    v_update = v_update.unsqueeze(1)
            
            
                # v_delta_check = v_update - v # shape = [32000, 1]
                # v_delta_check = v_delta_check.squeeze(1) # shape = [32000]

                # print('--- implicit tokens and their changed probabilities ---')
                # for k in tgt_toks:
                #     print(mt.tokenizer.decode(k), v_delta_check[k])

                # v_delta_sort, idx = torch.sort(v_delta_check, descending=False)

                # print('--- top 5 changed probabilities ---')
                # for i in range(20):
                #     print(mt.tokenizer.decode(idx[i]), v_delta_sort[i])
                
                # print('--- explicit tokens and their changed probabilities ---')
                # for k in explicit_toks:
                #     for i in range(len(idx)):
                #         if idx[i] == k:
                #             print(mt.tokenizer.decode(k), v_delta_check[k], 'rank=', i)

                v_delta_recheck = v - v_update # shape = [32000, 1]
                v_delta_2 = torch.zeros_like(v_delta_recheck)
                for i in explicit_toks:
                    v_delta_2[i] = rc_strength * v_delta_recheck[i]
                    print(mt.tokenizer.decode(i), v_delta_recheck[i]) # influence on explicit toks brought by interventing on implicit toks
                delta_h_2 = torch.matmul(A, v_delta_2.double()).squeeze(1)
                h[0, t] += delta_h_2.half()


                v_update = torch.matmul(W, h[0, t]) # v_update.shape = [32000, 1] or [32000]
                if len(v_update.shape) == 1:
                    v_update = v_update.unsqueeze(1)
                v_delta_check = torch.softmax(v_update, dim=0) - torch.softmax(v, dim=0) # shape = [32000, 1]
                v_delta_check = v_delta_check.squeeze(1) # shape = [32000]

                print('--- implicit tokens and their changed probabilities ---')
                for k in tgt_toks:
                    print(mt.tokenizer.decode(k), v_delta_check[k])

                v_delta_sort, idx = torch.sort(v_delta_check, descending=False)

                print('--- top 5 changed probabilities ---')
                for i in range(20):
                    print(mt.tokenizer.decode(idx[i]), v_delta_sort[i])
                
                print('--- explicit tokens and their changed probabilities ---')
                for k in explicit_toks:
                    for i in range(len(idx)):
                        if idx[i] == k:
                            print(mt.tokenizer.decode(k), v_delta_check[k], 'rank=', i)

                # raise Exception('debug')
            '''

        return x

    # With the patching rules defined, run the patched model in inference.

    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
    assert outputs_exp.logits.shape[0] == 2 # exp, compare
    # We report softmax probabilities for the answers_t token predictions of interest.
    '''
    debias_logits = torch.softmax(outputs_exp.logits[0, -1, :], dim=0).tolist()
    compare_logits = torch.softmax(outputs_exp.logits[1, -1, :], dim=0).tolist()
    debias_prob = 0
    compare_prob = 0

    for tok in explicit_toks:
        # print(tok)
        # print(debias_logits[tok], compare_logits[tok])
        debias_prob += debias_logits[tok]
        compare_prob += compare_logits[tok]

    debias_prob /= len(explicit_toks)
    compare_prob /= len(explicit_toks)

    
    debias_rank = get_rank(debias_logits, explicit_toks)
    compare_rank = get_rank(compare_logits, explicit_toks)

        
    return debias_prob, debias_rank, compare_prob, compare_rank
    '''
    return

def trace_with_patch_list(
    model,  # The model
    inp,  # A set of inputs
    inp_rephrase,
    states_to_patch,  # A list of (token index, layername) triples to restore
    answer_toks_list, 
    answer_list,
    prompt,
    prompt_rephrase,
    state,
    addi,
    addi_tok,
    other_model=None
    ):

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    x_before_list = ["model.layers.14"]


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    

    def patch_save(x, layer):
        global last_token_representation
        global last_token_representation_gen

        if layer not in save_layer_list:
            return x
        
        if last_token_representation is None:
            print(x[0].shape)
            last_token_representation = x[0][-1]
        if last_token_representation_gen is None:
            last_token_representation_gen = x[0][-1]
        return x
        

    def patch_resubmit(x, layer):
        global t_global_hopping
        global t_global_change_hopping

        global last_token_representation
        global last_token_representation_resubmit
        
        if layer in patch_spec:
            x_copy = copy.deepcopy(x)
            x_copy = untuple(x_copy)
            norm_x = Norm(x_copy.to(Norm_W.device))
            h = untuple(norm_x)
            if t_global_hopping is None:
                t_global_hopping = h.shape[1]-1
                t_global_temp_hopping = t_global_hopping
                t_global_change_hopping = 0
            else:
                t_global_change_hopping = t_global_change_hopping + 1
                t_global_temp_hopping = t_global_hopping + int(t_global_change_hopping // 32)

            t = h.shape[1]-1
            out = h[0, t].to(W.device)
            v = torch.matmul(W, out) # v.shape = [32000, 1] or [32000]?
            if len(v.shape) == 1:
                v = v.unsqueeze(1) # v.shape = [32000, 1]
                
            v = v.squeeze()
            logits = torch.softmax(v, dim=0).tolist()

            if state == "pre":
                #print("phmphmphm")
                answer_list_1 = copy.deepcopy(addi)
                answer_list_1.extend(answer_list)
                answer_toks_list_1 = copy.deepcopy(addi_tok)
                answer_toks_list_1.extend(answer_toks_list)
                #print(answer_list_1)
                #print(answer_toks_list_1)
                #print("phm2phm2phm2")
                '''
                answer_list_1 = answer_list
                answer_toks_list_1 = answer_toks_list
                '''
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
                if answer_string not in multi_results[state]["hopping"]["prob"].keys():
                    multi_results[state]["hopping"]["prob"][answer_string] = dict()
                    multi_results[state]["hopping"]["rank"][answer_string] = dict()

                if str(t_global_temp_hopping) in multi_results[state]["hopping"]["prob"][answer_string].keys():
                    multi_results[state]["hopping"]["prob"][answer_string][str(t_global_temp_hopping)].append(prob)
                    multi_results[state]["hopping"]["rank"][answer_string][str(t_global_temp_hopping)].append(rank)
                else:
                    multi_results[state]["hopping"]["prob"][answer_string][str(t_global_temp_hopping)] = []
                    multi_results[state]["hopping"]["rank"][answer_string][str(t_global_temp_hopping)] = []
                    multi_results[state]["hopping"]["prob"][answer_string][str(t_global_temp_hopping)].append(prob)
                    multi_results[state]["hopping"]["rank"][answer_string][str(t_global_temp_hopping)].append(rank)
        
        if layer in save_layer_list:
            if last_token_representation_resubmit is None:
                last_token_representation_resubmit = x[0][-1]
            return x

        if layer not in resubmit_layer_list:
            return x
        if last_token_representation is not None:
            #print(x[0].shape)
            x[0][-1] = 0 * x[0][-1] + last_token_representation.to(x.device)
            last_token_representation = None
        return x
    
    def patch_resubmit_resubmit(x, layer):
        global last_token_representation_resubmit

        if layer not in resubmit_layer_list:
            return x
        if last_token_representation_resubmit is not None:
            #print(x[0].shape)
            x[0][-1] = 0 * x[0][-1] + last_token_representation_resubmit.to(x.device)
            last_token_representation_resubmit = None
        return x
    
    def patch_resubmit_gen(x, layer):
        global last_token_representation_gen
        if x.shape[1] == 1:
            if layer in save_layer_list:
                #print("cnmcnmcnm:{}".format(x.shape))
                last_token_representation_gen = x[0][-1]
                return x
            if layer not in resubmit_layer_list:
                return x
            if last_token_representation_gen is not None:
                x[0][-1] = 0 * x[0][-1] + last_token_representation_gen.to(x.device)
        else:
            return x
        return x


    # Define the model-patching rule.
    def patch_rep(x, layer):
        global port_type
        global port_results
        global t_global
        global t_global_change
        global x_before
        '''
        tgt_toks = [tok_enc_1, tok_enc_2, ...]
        '''
        if layer not in patch_spec:
            return x
        
        if layer in x_before_list and state != "pre":
            other_model.model.base_model.set_x_before(untuple(x).detach().clone())
        #print(layer)
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        #h = untuple(x) # h.ahspe = [bs, seq_len, hid_dim]
        x_copy = copy.deepcopy(x)
        x_copy = untuple(x_copy)
        norm_x = Norm(x_copy.to(Norm_W.device))
        h = untuple(norm_x)
        #for t in patch_spec[layer]:
            # t is traced tok
        #print(h.shape[1])
        if t_global is None:
            t_global = h.shape[1]-1
            t_global_temp = t_global
            t_global_change = 0
        else:
            t_global_change = t_global_change + 1
            t_global_temp = t_global + int(t_global_change // 32)
        #print(t_global)
        #print(t_global_change)
               
        t = h.shape[1]-1
        out = h[0, t].to(W.device)
        v = torch.matmul(W, out) # v.shape = [32000, 1] or [32000]?
        if len(v.shape) == 1:
            v = v.unsqueeze(1) # v.shape = [32000, 1]
            
        v = v.squeeze()
        logits = torch.softmax(v, dim=0).tolist()

        
        if state == "pre":
            
            #print("phmphmphm")
            answer_list_1 = copy.deepcopy(addi)
            answer_list_1.extend(answer_list)
            answer_toks_list_1 = copy.deepcopy(addi_tok)
            answer_toks_list_1.extend(answer_toks_list)
            #print(answer_list_1)
            #print(answer_toks_list_1)
            #print("phm2phm2phm2")
            '''
            answer_list_1 = answer_list
            answer_toks_list_1 = answer_toks_list
            '''
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

    if state == "pre":
        '''
        outputs_exp_temp = model.generate(
            input_ids=inp['input_ids'],
            attention_mask=inp['attention_mask'],
            #min_new_tokens=len(target_new_tokens),
            max_new_tokens=20,
            #max_new_tokens=100, 
            do_sample=False,
            temperature=None,
            #num_beams=3,
        )
        '''
        outputs_exp_temp, _ = generate_with_cache(
            model,
            prompt_tokens=inp['input_ids'],
            max_gen_len=20,
            attention_mask=inp['attention_mask'],
            other_model = None,
            optimization=False
        )
        
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
            #print(gen_text_temp)
            #print(answer)
            #print(position)
            if position != 10000:
                toks = char_tok_dict_temp[position]
                toks_before = char_tok_dict_temp[position_before]
                answer_toks = outputs_exp_temp.detach().cpu().numpy().tolist()[0][toks_before:toks+1]
                    
                #print(outputs_exp_temp)
                #print(toks_before)
                #print(toks)
                #print(outputs_exp_temp[toks_before])
                #print(answer_toks)
                answer_toks_list_new.append(answer_toks)
                answer_list_new.append(answer)
                answer_toks_position_list_new.append([toks_before, toks])

        answer_toks_list = answer_toks_list_new
        answer_list = answer_list_new  

    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()      
    
    #print(list(patch_spec.keys()))
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        '''
        outputs_exp = model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=20,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        '''
        outputs_exp, _ = generate_with_cache(
            model,
            prompt_tokens=inp['input_ids'],
            max_gen_len=20,
            attention_mask=inp['attention_mask'],
            other_model = other_model,
            optimization=True
        )
        global t_global
        global t_global_change
        t_global = None
        t_global_change = None

    if state != "pre":
        other_model.model.set_list()
    with torch.no_grad():
        outputs_rephrase, _ = generate_with_cache(
            model,
            prompt_tokens=inp_rephrase['input_ids'],
            max_gen_len=20,
            attention_mask=inp_rephrase['attention_mask'],
            other_model = other_model,
            optimization=True
        )
    
    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()

    save_layer_list = ["model.norm"]
    resubmit_layer_list = ["model.embed_tokens"]
    '''
    with torch.no_grad(), nethook.TraceDict(
        model,
        save_layer_list,
        edit_output=patch_save,
    ) as td:
        '''
        outputs_for_save = model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=20,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        '''
        outputs_for_save, _ = generate_with_cache(
            model,
            prompt_tokens=inp['input_ids'],
            max_gen_len=1,
            attention_mask=inp['attention_mask'],
            other_model = other_model,
            optimization=False
        )

    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()

    with torch.no_grad(), nethook.TraceDict(
        origin_model,
        list(set(save_layer_list + resubmit_layer_list + list(patch_spec.keys()))),
        edit_output=patch_resubmit,
    ) as td:
        '''
        outputs_for_resubmit = origin_model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=20,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        '''
        outputs_for_resubmit, _ = generate_with_cache(
            model,
            prompt_tokens=inp['input_ids'],
            max_gen_len=1,
            attention_mask=inp['attention_mask'],
            other_model = other_model,
            optimization=False
        )
        global t_global_hopping
        global t_global_change_hopping
        t_global_hopping = None
        t_global_change_hopping = None

    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()
    
    with torch.no_grad(), nethook.TraceDict(
        origin_model,
        resubmit_layer_list,
        edit_output=patch_resubmit_resubmit,
    ) as td:
        '''
        outputs_for_resubmit_resubmit = origin_model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=20,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        '''
        outputs_for_resubmit_resubmit, _ = generate_with_cache(
            model,
            prompt_tokens=inp['input_ids'],
            max_gen_len=1,
            attention_mask=inp['attention_mask'],
            other_model = other_model,
            optimization=False
        )

    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()

    resubmit_layer_list = ["model.embed_tokens"]
    #resubmit_layer_list = ["model.norm"]
    with torch.no_grad(), nethook.TraceDict(
        model,
        save_layer_list + resubmit_layer_list,
        edit_output=patch_resubmit_gen,
    ) as td:
        '''
        outputs_for_resubmit_gen = model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=20,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        '''
        outputs_for_resubmit_gen, _ = generate_with_cache(
            model,
            prompt_tokens=inp['input_ids'],
            max_gen_len=1,
            attention_mask=inp['attention_mask'],
            other_model = other_model,
            optimization=False
        )
        
    
    
    #out_test = model.forward(inp['input_ids'], output_attentions=True)
    #print(len(out_test.attentions))
    #print(out_test.attentions[0].shape)
    '''
    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()
    
    outputs_for_attentionmap, cache_for_attentionmap = generate_with_cache(
        model,
        prompt_tokens=inp['input_ids'],
        #attention_mask=inp['attention_mask'],
        #min_new_tokens=len(target_new_tokens),
        max_gen_len=1,
        attention_mask=inp['attention_mask'],
        other_model = other_model,
        optimization=False
        #max_new_tokens=100, 
        #do_sample=False,
        #temperature=None
        #tokenizer=mt.tokenizer
    )
    
    if state != "pre":   
        other_model.model.set_list()
        other_model.model.set_attention()
    #assert outputs_exp.shape[0] == 1 # exp, compare
    # We report softmax probabilities for the answers_t token predictions of interest.
    if state == "pre":
        return outputs_exp, outputs_rephrase, answer_toks_list, answer_list, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen, cache_for_attentionmap, answer_toks_position_list_new
    else:
        return outputs_exp, outputs_rephrase, answer_toks_list, answer_list, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen, cache_for_attentionmap



def generate_with_cache(
    model,
    prompt_tokens,
    max_gen_len,
    attention_mask,
    other_model,
    optimization
):
    
        kvcache = None
        cache_list = []
        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = max_gen_len + max_prompt_len
        pad_id = mt.tokenizer.pad_token_id
        #print(pad_id)
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
            #TODO: logits = forward(tokens, prev_pos)
            out = model.forward(tokens, prev_pos, output_attentions=True, attention_mask=attention_mask_all)
            cache_list.append(out.attentions)
            
        stop_tokens = torch.tensor(list([mt.tokenizer.eos_token_id]), device="cuda")

        
        #min_prompt_len-1意味着最后一个input token是单独推理的
        for cur_pos in range(min_prompt_len-1, total_len):
            #print(cur_pos)
            #TODO: logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if other_model is not None:
                attention_mask_list = other_model.model.get_attention()
                if attention_mask_list is not None:
                    attention_mask_all[:, attention_mask_list[0]:attention_mask_list[1]] = 0
            if prev_pos == 0:
                with torch.no_grad():
                    out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos])   
                    if other_model is not None:
                        for i in range(len(other_model.lora_list)):
                            other_model.model.get_submodule(other_model.lora_list[i]).initialize_lora_weights()
            else:
                if other_model is not None:
                    optimizer = torch.optim.Adam(other_model.decode_optim_parameters(), 0.01)
                    for e_g in range(10):
                        out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos]) 
                        loss = other_model.model.get_decode_loss()
                        print(other_model.print_decode_optim_parameters())
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                with torch.no_grad():
                    out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True, attention_mask=attention_mask_all[:, 0:cur_pos]) 
            #print(out)
            kvcache = out.past_key_values
            cache_list.append(out.attentions)

            #print()
            next_token = torch.argmax(out.logits[:, -1], dim=-1)


            next_token[1] = next_token[0]
            # only replace token if prompt has already been generated
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
            # cut to max gen len
            start = 0 
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            # cut to after eos tok if any
            for stop_token in [mt.tokenizer.eos_token_id]:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return tokens, cache_list


def debias_states(model, num_layers, inp, check_tok_ids, implicit_toks, explicit_toks):
    #print("gggg:{}".format(check_tok_ids))
    #print("bbbb:{}".format(num_layers))
    for tnum in check_tok_ids:
        for layer in range(0, num_layers):
            trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                implicit_toks,
                explicit_toks
            )
            
    return

def debias_states_list(editor, model, num_layers, inp, test_inp, test_inp_rephrase, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list, prompt, prompt_test, prompt_test_rephrase, subject, relation, target_new, target_new_tok, subject_tok, relation_tok, answer_tok):
    global max_attention_dict
    
    states_to_patch = []
    for tnum in test_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(model, layer)))


    outputs_exp, outputs_rephrase, answer_toks_list, answer_list, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen, cache_for_attentionmap, answer_toks_position_list_new = trace_with_patch_list(
        model,
        test_inp,
        test_inp_rephrase,
        states_to_patch,
        answer_toks_list,
        answer_list,
        prompt_test,
        prompt_test_rephrase,
        state="pre",
        addi=target_new[0:2],
        addi_tok=target_new_tok[0:2]
    )

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
 
    #print("phmphmphm")
    assert isinstance(target_new, list)
    target_new_temp = list()
    target_new_tok_temp = list()
    for edit_index in range(len(target_new)):
        if edit_index >= 2:
            break
        edited_model = editor.simple_edit(
            prompts=[prompt],
            target_new=[target_new[edit_index]],
            subject_tok=subject_tok,
            relation_tok=relation_tok,
            answer_tok=answer_tok,
            subject=[subject],
            mem_requests=memory_prompts
            #train_ds=None,
            #keep_original_weight=False,
            #pre_file=args.pre_file,
            #pre_edit = pre_edit,
            #test_generation=False,
            #edit_type = edit_type,
            #origin_edit_id = origin_edit_id
            )
        editor.model = edited_model
        target_new_temp.append(target_new[edit_index])
        target_new_tok_temp.append(target_new_tok[edit_index])

    
    '''
    for key in edited_model.model.model.state_dict().keys():
        print(key)
    '''

    states_to_patch = []
    for tnum in test_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(edited_model.model.model, layer)))
    
    outputs_exp_post, outputs_rephrase_post, answer_toks_list_all, answer_list_all, outputs_for_save_post, outputs_for_resubmit_post, outputs_for_resubmit_resubmit_post, outputs_for_resubmit_gen_post, cache_for_attentionmap_post = trace_with_patch_list(
        #edited_model.model,
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
        other_model=edited_model
    )
    
    gen_text = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_exp.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1 = ''
    char_tok_dict = dict()
    j = 0
    for t in range(len(gen_text)):
        
        t_1 = gen_text[t]
        for i in range(len(t_1)):
            gen_text_1 = gen_text_1 + t_1[i]
            char_tok_dict[j] = t
            j = j + 1
    
    gen_text_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_exp_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_post = ''
    char_tok_dict_post = dict()
    j = 0
    for t in range(len(gen_text_post)):
        
        t_1 = gen_text_post[t]
        for i in range(len(t_1)):
            gen_text_1_post = gen_text_1_post + t_1[i]
            char_tok_dict_post[j] = t
            j = j + 1

    gen_text_rephrase_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_rephrase_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_rephrase_post = ''
    char_tok_dict_rephrase_post = dict()
    j = 0
    for t in range(len(gen_text_rephrase_post)):
        
        t_1 = gen_text_rephrase_post[t]
        for i in range(len(t_1)):
            gen_text_1_rephrase_post = gen_text_1_rephrase_post + t_1[i]
            char_tok_dict_rephrase_post[j] = t
            j = j + 1

    gen_text_save = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_save.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_save = ''
    char_tok_dict_save = dict()
    j = 0
    for t in range(len(gen_text_save)):
        
        t_1 = gen_text_save[t]
        for i in range(len(t_1)):
            gen_text_1_save = gen_text_1_save + t_1[i]
            char_tok_dict_save[j] = t
            j = j + 1

    gen_text_save_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_save_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_save_post = ''
    char_tok_dict_save_post = dict()
    j = 0
    for t in range(len(gen_text_save_post)):
        
        t_1 = gen_text_save_post[t]
        for i in range(len(t_1)):
            gen_text_1_save_post = gen_text_1_save_post + t_1[i]
            char_tok_dict_save_post[j] = t
            j = j + 1

    gen_text_resubmit = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit = ''
    char_tok_dict_resubmit = dict()
    j = 0
    for t in range(len(gen_text_resubmit)):
        
        t_1 = gen_text_resubmit[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit = gen_text_1_resubmit + t_1[i]
            char_tok_dict_resubmit[j] = t
            j = j + 1

    gen_text_resubmit_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_post = ''
    char_tok_dict_resubmit_post = dict()
    j = 0
    for t in range(len(gen_text_resubmit_post)):
        
        t_1 = gen_text_resubmit_post[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_post = gen_text_1_resubmit_post + t_1[i]
            char_tok_dict_resubmit_post[j] = t
            j = j + 1



    gen_text_resubmit_resubmit = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_resubmit.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_resubmit = ''
    char_tok_dict_resubmit_resubmit = dict()
    j = 0
    for t in range(len(gen_text_resubmit_resubmit)):
        
        t_1 = gen_text_resubmit_resubmit[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_resubmit = gen_text_1_resubmit_resubmit + t_1[i]
            char_tok_dict_resubmit_resubmit[j] = t
            j = j + 1

    gen_text_resubmit_resubmit_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_resubmit_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_resubmit_post = ''
    char_tok_dict_resubmit_resubmit_post = dict()
    j = 0
    for t in range(len(gen_text_resubmit_resubmit_post)):
        
        t_1 = gen_text_resubmit_resubmit_post[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_resubmit_post = gen_text_1_resubmit_resubmit_post + t_1[i]
            char_tok_dict_resubmit_resubmit_post[j] = t
            j = j + 1

    gen_text_resubmit_gen = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_gen.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_gen = ''
    char_tok_dict_resubmit_gen = dict()
    j = 0
    for t in range(len(gen_text_resubmit_gen)):
        
        t_1 = gen_text_resubmit_gen[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_gen = gen_text_1_resubmit_gen + t_1[i]
            char_tok_dict_resubmit_gen[j] = t
            j = j + 1

    gen_text_resubmit_gen_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_gen_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_gen_post = ''
    char_tok_dict_resubmit_gen_post = dict()
    j = 0
    for t in range(len(gen_text_resubmit_gen_post)):
        
        t_1 = gen_text_resubmit_gen_post[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_gen_post = gen_text_1_resubmit_gen_post + t_1[i]
            char_tok_dict_resubmit_gen_post[j] = t
            j = j + 1

    subject_position, subject_position_before = find_sublist_position_inprompt(prompt_test, subject)
    subject_tok, subject_tok_before = char_tok_dict[subject_position], char_tok_dict[subject_position_before]

    relation_position, relation_position_before = find_sublist_position_inprompt(prompt_test, relation)
    relation_tok, relation_tok_before = char_tok_dict[relation_position], char_tok_dict[relation_position_before]
    
    context_position, context_position_before = find_sublist_position_context(prompt_test, "Q:")
    context_tok, context_tok_before = char_tok_dict[context_position], char_tok_dict[context_position_before]
    
    multi_results["pre"]["generate"].append(
        {
            "prompt": prompt_test,
            "gen_tok": outputs_exp.detach().cpu().numpy().tolist()[0],
            "gen_text": gen_text_1,
            "gen_text_save": gen_text_1_save,
            "gen_text_resubmit": gen_text_1_resubmit,
            "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit,
            "gen_text_resubmit_gen": gen_text_1_resubmit_gen,
            "answer_list_all": answer_list_all,
            "answer_list_success": answer_list,
            "answer_list_miss": target_new_temp,
            #"answer_toks_list": answer_toks_list,
            "char_tok_dict": char_tok_dict,
            "char_tok_dict_resubmit": char_tok_dict_resubmit,
            "insert_tok": str(test_inp["input_ids"].shape[1]-1),
            "subject_tok": [subject_tok, subject_tok_before],
            "relation_tok": [relation_tok, relation_tok_before],
            "context_tok": [context_tok, context_tok_before]
        }
    )
    
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
            #"answer_toks_list": answer_toks_list_post,
            "char_tok_dict": char_tok_dict_post,
            "char_tok_dict_resubmit": char_tok_dict_resubmit_post,
            "insert_tok": str(test_inp["input_ids"].shape[1]-1),
            "subject_tok": [subject_tok, subject_tok_before],
            "relation_tok": [relation_tok, relation_tok_before],
            "context_tok": [context_tok, context_tok_before]
        }
    )

    begin_tok = inp["input_ids"].shape[1]-1
    for l in range(32):    
        cache = cache_for_attentionmap[0]
        cache_post = cache_for_attentionmap_post[0]
        attention_pattern = cache[l][0]
        attention_pattern_post = cache_post[l][0]
        print("the shape of attention is: {}".format(attention_pattern.shape))
        #imshow(attention_pattern, layer=l, state="pre", gen=0)
        #imshow(attention_pattern_post, layer=l, state="post", gen=0)
        for h in range(32):
            
            attention_row = attention_pattern[h][-1]
            attention_row[0:context_tok_before] = -float('inf')
            _, idx = attention_row.topk(1)
            if idx[0] >= subject_tok_before and idx[0] <= subject_tok:
                max_attention_dict[f"layer_{l}_head_{h}"]["subject"] += 1
                max_attention_dict[f"layer_{l}"]["subject"] += 1
            elif idx[0] >= relation_tok_before and idx[0] <= relation_tok:
                max_attention_dict[f"layer_{l}_head_{h}"]["relation"] += 1
                max_attention_dict[f"layer_{l}"]["relation"] += 1
            elif idx[0] < context_tok_before:
                max_attention_dict[f"layer_{l}_head_{h}"]["context"] += 1
                max_attention_dict[f"layer_{l}"]["context"] += 1

            attention_row_post = attention_pattern_post[h][-1]
            attention_row_post[0:context_tok_before] = -float('inf')
            _, idx_post = attention_row_post.topk(1)
            if idx_post[0] >= subject_tok_before and idx_post[0] <= subject_tok:
                max_attention_dict_post[f"layer_{l}_head_{h}"]["subject"] += 1
                max_attention_dict_post[f"layer_{l}"]["subject"] += 1
            elif idx_post[0] >= relation_tok_before and idx_post[0] <= relation_tok:
                max_attention_dict_post[f"layer_{l}_head_{h}"]["relation"] += 1
                max_attention_dict_post[f"layer_{l}"]["relation"] += 1
            elif idx_post[0] < context_tok_before:
                max_attention_dict_post[f"layer_{l}_head_{h}"]["context"] += 1
                max_attention_dict_post[f"layer_{l}"]["context"] += 1
            
            '''
            attention_row = attention_pattern[h][-1]
            _, idx = attention_row.topk(5)
            for idx_i in idx:
                if idx_i >= subject_tok_before and idx_i <= subject_tok:
                    max_attention_dict[f"layer_{l}_head_{h}"]["subject"] += 1
                    max_attention_dict[f"layer_{l}"]["subject"] += 1
                    break
            for idx_i in idx:
                if idx_i >= relation_tok_before and idx_i <= relation_tok:
                    max_attention_dict[f"layer_{l}_head_{h}"]["relation"] += 1
                    max_attention_dict[f"layer_{l}"]["relation"] += 1
                    break
            for idx_i in idx:
                if idx_i < context_tok_before:
                    max_attention_dict[f"layer_{l}_head_{h}"]["context"] += 1
                    max_attention_dict[f"layer_{l}"]["context"] += 1
                    break

            attention_row_post = attention_pattern_post[h][-1]
            _, idx_post = attention_row_post.topk(5)
            for idx_post_i in idx_post:
                if idx_post_i >= subject_tok_before and idx_post_i <= subject_tok:
                    max_attention_dict_post[f"layer_{l}_head_{h}"]["subject"] += 1
                    max_attention_dict_post[f"layer_{l}"]["subject"] += 1
                    break
            for idx_post_i in idx_post:
                if idx_post_i >= relation_tok_before and idx_post_i <= relation_tok:
                    max_attention_dict_post[f"layer_{l}_head_{h}"]["relation"] += 1
                    max_attention_dict_post[f"layer_{l}"]["relation"] += 1
                    break
            for idx_post_i in idx_post:
                if idx_post_i < context_tok_before:
                    max_attention_dict_post[f"layer_{l}_head_{h}"]["context"] += 1
                    max_attention_dict_post[f"layer_{l}"]["context"] += 1
                    break   
            '''
        #TODO:单个样本decode过程（算context,不算context，看看有什么不同和相同）
        #cache = cache_for_attentionmap[0]
        #attention_pattern = cache[l][0]
        '''
        if len(answer_toks_position_list_new) >= 3:
            print("one sample!!!!!!!!!!!!!!!")
            for p in range(len(answer_toks_position_list_new)):
                if p >= 3:
                    break
                t_i = answer_toks_position_list_new[p][0] - begin_tok
                while t_i <= (answer_toks_position_list_new[p][1] - begin_tok):
                    cache_multi = cache_for_attentionmap[t_i]
                    attention_pattern_multi = cache_multi[l][0]
                    for h in range(32):
                        attention_row_multi = attention_pattern_multi[h][-1]
                        
                        _, idx_multi = attention_row_multi.topk(1)
                        if idx_multi[0] >= subject_tok_before and idx_multi[0] <= subject_tok:
                            max_attention_dict_multi[f"layer_{l}_head_{h}_position_{p}"]["subject"] += 1
                            max_attention_dict_multi[f"layer_{l}_position_{p}"]["subject"] += 1
                        elif idx_multi[0] >= relation_tok_before and idx_multi[0] <= relation_tok:
                            max_attention_dict_multi[f"layer_{l}_head_{h}_position_{p}"]["relation"] += 1
                            max_attention_dict_multi[f"layer_{l}_position_{p}"]["relation"] += 1
                        elif idx_multi[0] <= context_tok_before:
                            max_attention_dict_multi[f"layer_{l}_head_{h}_position_{p}"]["context"] += 1
                            max_attention_dict_multi[f"layer_{l}_position_{p}"]["context"] += 1
                        elif idx_multi[0] >= answer_toks_position_list_new[0][0] and idx_multi[0] <= answer_toks_position_list_new[0][1]:
                            max_attention_dict_multi[f"layer_{l}_head_{h}_position_{p}"]["answer0"] += 1
                            max_attention_dict_multi[f"layer_{l}_position_{p}"]["answer0"] += 1
                        elif idx_multi[0] >= answer_toks_position_list_new[1][0] and idx_multi[0] <= answer_toks_position_list_new[1][1]:
                            max_attention_dict_multi[f"layer_{l}_head_{h}_position_{p}"]["answer1"] += 1
                            max_attention_dict_multi[f"layer_{l}_position_{p}"]["answer1"] += 1
                        elif idx_multi[0] >= answer_toks_position_list_new[2][0] and idx_multi[0] <= answer_toks_position_list_new[2][1]:
                            max_attention_dict_multi[f"layer_{l}_head_{h}_position_{p}"]["answer2"] += 1
                            max_attention_dict_multi[f"layer_{l}_position_{p}"]["answer2"] += 1
                    t_i = t_i + 1
        '''
    print("************* attention **************")
    print(max_attention_dict)
    print("************* attention post **************")
    print(max_attention_dict_post)
    print("************* total number **************")
    print(len(multi_results["pre"]["generate"]))
    return

def debias_states_list_port(editor, model, num_layers, hop1_inp, hop2_inp, reasoning_inp, hop1_check_tok_ids, hop2_check_tok_ids, reasoning_check_tok_ids, hop1_prompt, hop2_prompt, reasoning_prompt, implicit_answer, explicit_answer, explicit_answer_2, implicit_toks, explicit_toks, explicit_toks_2, hop1_subject_tok, hop1_relation_tok, hop1_answer_tok, hop2_subject_tok, hop2_relation_tok, hop2_answer_tok):
    states_to_patch = []
    for tnum in reasoning_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(model, layer)))

    outputs_exp, explicit_toks_nouse, explicit_answer_nouse, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen, cache_for_attentionmap, answer_toks_position_list_new = trace_with_patch_list(
        model,
        reasoning_inp,
        states_to_patch,
        explicit_toks_2,
        explicit_answer_2,
        reasoning_prompt,
        state="pre",
        addi=list(),
        addi_tok=list()
    )

    edited_model = editor.simple_edit(
        prompts=[hop1_prompt],
        target_new=implicit_answer,
        subject_tok=hop1_subject_tok,
        relation_tok=hop1_relation_tok,
        answer_tok=hop1_answer_tok,
        subject=[""],
        mem_requests=memory_prompts,
        #train_ds=None,
        #keep_original_weight=False,
        #pre_file=args.pre_file,
        #pre_edit = pre_edit,
        #test_generation=False,
        #edit_type = edit_type,
        #origin_edit_id = origin_edit_id
        )
    editor.model = edited_model

    
    edited_model = editor.simple_edit(
        prompts=[hop2_prompt],
        target_new=explicit_answer,
        subject_tok=hop2_subject_tok,
        relation_tok=hop2_relation_tok,
        answer_tok=hop2_answer_tok,
        subject=[""],
        mem_requests=memory_prompts,
        #train_ds=None,
        #keep_original_weight=False,
        #pre_file=args.pre_file,
        #pre_edit = pre_edit,
        #test_generation=False,
        #edit_type = edit_type,
        #origin_edit_id = origin_edit_id
        )
    editor.model = edited_model

    states_to_patch = []
    for tnum in reasoning_check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(edited_model.model.model, layer)))
    
    outputs_exp_post, explicit_toks_nouse, explicit_answer_nouse, outputs_for_save_post, outputs_for_resubmit_post, outputs_for_resubmit_resubmit_post, outputs_for_resubmit_gen_post, cache_for_attentionmap_post = trace_with_patch_list(
        edited_model.model.model,
        reasoning_inp,
        states_to_patch,
        explicit_toks_2,
        explicit_answer_2,
        reasoning_prompt,
        state="post",
        addi=list(),
        addi_tok=list(),
        other_model=edited_model
    )

    begin_decode_tok = edited_model.model.base_model.get_begin_decode_tok()
    outputs_exp_post = outputs_exp_post[:, reasoning_check_tok_ids[0]+begin_decode_tok:]

    gen_text = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_exp.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1 = ''
    char_tok_dict = dict()
    j = 0
    for t in range(len(gen_text)):
        
        t_1 = gen_text[t]
        for i in range(len(t_1)):
            gen_text_1 = gen_text_1 + t_1[i]
            char_tok_dict[j] = t
            j = j + 1
    
    gen_text_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_exp_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_post = ''
    char_tok_dict_post = dict()
    j = 0
    for t in range(len(gen_text_post)):
        
        t_1 = gen_text_post[t]
        for i in range(len(t_1)):
            gen_text_1_post = gen_text_1_post + t_1[i]
            char_tok_dict_post[j] = t
            j = j + 1


    gen_text_save = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_save.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_save = ''
    char_tok_dict_save = dict()
    j = 0
    for t in range(len(gen_text_save)):
        
        t_1 = gen_text_save[t]
        for i in range(len(t_1)):
            gen_text_1_save = gen_text_1_save + t_1[i]
            char_tok_dict_save[j] = t
            j = j + 1

    gen_text_save_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_save_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_save_post = ''
    char_tok_dict_save_post = dict()
    j = 0
    for t in range(len(gen_text_save_post)):
        
        t_1 = gen_text_save_post[t]
        for i in range(len(t_1)):
            gen_text_1_save_post = gen_text_1_save_post + t_1[i]
            char_tok_dict_save_post[j] = t
            j = j + 1

    gen_text_resubmit = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit = ''
    char_tok_dict_resubmit = dict()
    j = 0
    for t in range(len(gen_text_resubmit)):
        
        t_1 = gen_text_resubmit[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit = gen_text_1_resubmit + t_1[i]
            char_tok_dict_resubmit[j] = t
            j = j + 1

    gen_text_resubmit_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_post = ''
    char_tok_dict_resubmit_post = dict()
    j = 0
    for t in range(len(gen_text_resubmit_post)):
        
        t_1 = gen_text_resubmit_post[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_post = gen_text_1_resubmit_post + t_1[i]
            char_tok_dict_resubmit_post[j] = t
            j = j + 1



    gen_text_resubmit_resubmit = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_resubmit.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_resubmit = ''
    char_tok_dict_resubmit_resubmit = dict()
    j = 0
    for t in range(len(gen_text_resubmit_resubmit)):
        
        t_1 = gen_text_resubmit_resubmit[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_resubmit = gen_text_1_resubmit_resubmit + t_1[i]
            char_tok_dict_resubmit_resubmit[j] = t
            j = j + 1

    gen_text_resubmit_resubmit_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_resubmit_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_resubmit_post = ''
    char_tok_dict_resubmit_resubmit_post = dict()
    j = 0
    for t in range(len(gen_text_resubmit_resubmit_post)):
        
        t_1 = gen_text_resubmit_resubmit_post[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_resubmit_post = gen_text_1_resubmit_resubmit_post + t_1[i]
            char_tok_dict_resubmit_resubmit_post[j] = t
            j = j + 1

    gen_text_resubmit_gen = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_gen.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_gen = ''
    char_tok_dict_resubmit_gen = dict()
    j = 0
    for t in range(len(gen_text_resubmit_gen)):
        
        t_1 = gen_text_resubmit_gen[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_gen = gen_text_1_resubmit_gen + t_1[i]
            char_tok_dict_resubmit_gen[j] = t
            j = j + 1

    gen_text_resubmit_gen_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_resubmit_gen_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_resubmit_gen_post = ''
    char_tok_dict_resubmit_gen_post = dict()
    j = 0
    for t in range(len(gen_text_resubmit_gen_post)):
        
        t_1 = gen_text_resubmit_gen_post[t]
        for i in range(len(t_1)):
            gen_text_1_resubmit_gen_post = gen_text_1_resubmit_gen_post + t_1[i]
            char_tok_dict_resubmit_gen_post[j] = t
            j = j + 1

    #print(gen_text_1)
    #print(gen_text_1_post)

    multi_results["pre"]["generate"].append(
        {
            "prompt": reasoning_prompt,
            "gen_tok": outputs_exp.detach().cpu().numpy().tolist()[0],
            "gen_text": gen_text_1,
            "gen_text_save": gen_text_1_save,
            "gen_text_resubmit": gen_text_1_resubmit,
            "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit,
            "gen_text_resubmit_gen": gen_text_1_resubmit_gen,
            #"answer_list_all": answer_list_all,
            #"answer_list_success": answer_list,
            #"answer_list_miss": target_new_temp,
            #"answer_toks_list": answer_toks_list,
            "char_tok_dict": char_tok_dict,
            "char_tok_dict_resubmit": char_tok_dict_resubmit,
            "insert_tok": str(reasoning_inp["input_ids"].shape[1]-1)
        }
    )
    
    multi_results["post"]["generate"].append(
        {
            "prompt": reasoning_prompt,
            "gen_tok": outputs_exp_post.detach().cpu().numpy().tolist()[0],
            "gen_text": gen_text_1_post, 
            "gen_text_save": gen_text_1_save_post,
            "gen_text_resubmit": gen_text_1_resubmit_post,
            "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit_post,
            "gen_text_resubmit_gen": gen_text_1_resubmit_gen_post,
            #"answer_list_all": answer_list_all,
            #"answer_list_success": answer_list,
            #"answer_list_miss": target_new_temp,
            #"answer_toks_list": answer_toks_list_post,
            "char_tok_dict": char_tok_dict_post,
            "char_tok_dict_resubmit": char_tok_dict_resubmit_post,
            "insert_tok": str(reasoning_inp["input_ids"].shape[1]-1)
        }
    )

    edited_model.model.base_model.set_representation()
    return
        
def calculate_hidden_flow(
    mt, prompt, check_tok_ids, implicit_toks, explicit_toks,
    ):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * 2)
    with torch.no_grad():
        out = mt.model(**inp)["logits"] # out.shape = (2, seq_len, vocab_size)
        logits = torch.softmax(out[0, -1], dim=0).tolist() # len = vocab_size
        origin_prob = 0
        for tok in explicit_toks:
            origin_prob += logits[tok]
        origin_prob /= len(explicit_toks)
        #origin_rank = get_rank(logits, explicit_toks)

    debias_states(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks)
    
    return 

def calculate_hidden_generation_flow(
    editor, mt, prompt, prompt_test, prompt_test_rephrase, subject, relation, target_new, target_new_tok, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list
    ):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt])
    test_inp = make_inputs(mt.tokenizer, [prompt_test, prompt_test])
    test_inp_rephrase = make_inputs(mt.tokenizer, [prompt_test_rephrase, prompt_test_rephrase])

    token_list = mt.tokenizer.encode(prompt)
    gen_text = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in token_list]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1 = ''
    char_tok_dict = dict()
    j = 0
    for t in range(len(gen_text)):
        
        t_1 = gen_text[t]
        for i in range(len(t_1)):
            gen_text_1 = gen_text_1 + t_1[i]
            char_tok_dict[j] = t
            j = j + 1
    
    subject_position, subject_position_before = find_sublist_position_inprompt(prompt, subject)
    subject_tok, subject_tok_before = char_tok_dict[subject_position], char_tok_dict[subject_position_before]
    print(f"subject_tok: {gen_text[subject_tok]}")

    relation_position, relation_position_before = find_sublist_position_inprompt(prompt, relation)
    relation_tok, relation_tok_before = char_tok_dict[relation_position], char_tok_dict[relation_position_before]
    print(f"relation_tok: {gen_text[relation_tok]}")

    debias_states_list(editor, mt.model, mt.num_layers, inp, test_inp, test_inp_rephrase, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list, prompt, prompt_test, prompt_test_rephrase, subject, relation, target_new, target_new_tok, subject_tok, relation_tok, list())
    
    return 


def calculate_hidden_generation_flow_port(
    editor, mt, hop1_prompt, hop2_prompt, hop1_subject, hop2_subject, hop1_relation, hop2_relation, reasoning_prompt, reasoning_prompt_search, hop1_check_tok_ids, hop2_check_tok_ids, reasoning_check_tok_ids, implicit_answer, explicit_answer, explicit_answer_2, implicit_toks, explicit_toks, explicit_toks_2 
    ):
    hop1_inp = make_inputs(mt.tokenizer, [hop1_prompt])
    hop2_inp = make_inputs(mt.tokenizer, [hop2_prompt])
    reasoning_inp = make_inputs(mt.tokenizer, [reasoning_prompt, reasoning_prompt_search])

    hop1_token_list = mt.tokenizer.encode(hop1_prompt)
    hop2_token_list = mt.tokenizer.encode(hop2_prompt)
    hop1_gen_text = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in hop1_token_list]
    hop2_gen_text = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in hop2_token_list]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    hop1_gen_text_1 = ''
    hop1_char_tok_dict = dict()
    j = 0
    for t in range(len(hop1_gen_text)):
        
        t_1 = hop1_gen_text[t]
        for i in range(len(t_1)):
            hop1_gen_text_1 = hop1_gen_text_1 + t_1[i]
            hop1_char_tok_dict[j] = t
            j = j + 1

    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    hop2_gen_text_1 = ''
    hop2_char_tok_dict = dict()
    j = 0
    for t in range(len(hop2_gen_text)):
        
        t_1 = hop2_gen_text[t]
        for i in range(len(t_1)):
            hop2_gen_text_1 = hop2_gen_text_1 + t_1[i]
            hop2_char_tok_dict[j] = t
            j = j + 1

    hop1_subject_position, hop1_subject_position_before = find_sublist_position_inprompt(hop1_prompt, hop1_subject)
    hop1_subject_tok, hop1_subject_tok_before = hop1_char_tok_dict[hop1_subject_position], hop1_char_tok_dict[hop1_subject_position_before]
    print(f"hop1_subject_tok: {hop1_gen_text[hop1_subject_tok]}")

    hop1_relation_position, hop1_relation_position_before = find_sublist_position_inprompt(hop1_prompt, hop1_relation)
    hop1_relation_tok, hop1_relation_tok_before = hop1_char_tok_dict[hop1_relation_position], hop1_char_tok_dict[hop1_relation_position_before]
    print(f"hop1_relation_tok: {hop1_gen_text[hop1_relation_tok]}")

    hop2_subject_position, hop2_subject_position_before = find_sublist_position_inprompt(hop2_prompt, hop2_subject)
    hop2_subject_tok, hop2_subject_tok_before = hop2_char_tok_dict[hop2_subject_position], hop2_char_tok_dict[hop2_subject_position_before]
    print(f"hop2_subject_tok: {hop2_gen_text[hop2_subject_tok]}")

    hop2_relation_position, hop2_relation_position_before = find_sublist_position_inprompt(hop2_prompt, hop2_relation)
    hop2_relation_tok, hop2_relation_tok_before = hop2_char_tok_dict[hop2_relation_position], hop2_char_tok_dict[hop2_relation_position_before]
    print(f"hop2_relation_tok: {hop2_gen_text[hop2_relation_tok]}")
    
    debias_states_list_port(editor, mt.model, mt.num_layers, hop1_inp, hop2_inp, reasoning_inp, hop1_check_tok_ids, hop2_check_tok_ids, reasoning_check_tok_ids, hop1_prompt, hop2_prompt, reasoning_prompt, implicit_answer, explicit_answer, explicit_answer_2, implicit_toks, explicit_toks, explicit_toks_2, hop1_subject_tok, hop1_relation_tok, list(), hop2_subject_tok, hop2_relation_tok, list())
    return
'''
model_name = "/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B"
torch_dtype = torch.float32
device_map = 'balanced'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map, max_memory={0: "0GiB", 1: "20GiB", 2: "0GiB", 3: "0GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "31GiB"})
#{0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "31GiB", 7: "0GiB"}
#{0: "20GiB", 1: "31GiB", 2: "0GiB", 3: "31GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
'''

hparams = MELOHyperParams.from_hparams('/home/hmpiao/EasyEdit/hparams/MELO/llama3-8b.yaml')
#hparams = MELOHyperParams.from_hparams('/home/hmpiao/EasyEdit/hparams/TPATCHER/llama3-8b.yaml')
#hparams = GraceHyperParams.from_hparams('/home/hmpiao/EasyEdit/hparams/GRACE/llama3-8b.yaml')
editor = BaseEditor.from_hparams(hparams)

#tok = AutoTokenizer.from_pretrained(model_name)

mt = ModelAndTokenizer(
    tokenizer=editor.tok,
    #low_cpu_mem_usage=IS_COLAB,
    model=editor.model,
    torch_dtype = editor.torch_dtype,
    #model_type = model_type,
)
#print("cnmcnmcnm: {}".format(type(mt.model)))

vocab_size = int(mt.model.state_dict()['lm_head.weight'].shape[0])

#write_path = "/data2/hmpiao/FME/log_analysis/port.json"
#write_path = "/data2/hmpiao/FME/log_analysis/multinormtokedit_meloall_attention.json"
write_path = "/data2/hmpiao/FME/log_analysis/multi_analysis_all_continue.json"
#write_path = "/data2/hmpiao/FME/log_analysis/port_analysis_all_noloss_att_nocontext_faketest.json"
#write_path = "/data2/hmpiao/FME/log_analysis/sequential_analysis_all_noloss_att_nocontext_faketest.json"
#write_path = "/data2/hmpiao/FME/log_analysis/multinormtokedit_grace_attention.json"
#write_path = "/data2/hmpiao/FME/log_analysis/sequentialnormtokedit.json"

#data_path = "/home/hmpiao/EasyEdit/data/demo_port_analysis_all_rephrase.json"
data_path = "/home/hmpiao/EasyEdit/data/demo_multi_analysis_all_rephrase.json"
#data_path = "/home/hmpiao/EasyEdit/data/demo_base_dc_100.json"
#data_path = "/home/hmpiao/EasyEdit/data/demo_sequential_analysis.json"

memory_data_path = "/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json"

fr = open(data_path, "r")
data = json.load(fr)

memory_fr = open(memory_data_path, "r")
memory_data = json.load(memory_fr)
memory_data = random.sample(memory_data, 100)
memory_prompts = list()
for d in memory_data:
    if "locality" in d.keys():
        for k in d["locality"].keys():
            for p in d["locality"][k]:
                memory_prompts.append(p["prompt"])

#print(mt.model.state_dict())

for key in mt.model.state_dict().keys():
    print(key)
W = copy.deepcopy(mt.model.state_dict()['lm_head.weight'])
Norm_W = copy.deepcopy(mt.model.state_dict()['model.norm.weight'])
Norm = copy.deepcopy(mt.model.model.norm)
origin_model = copy.deepcopy(mt.model)

output_data = list()

#TODO: 在数据集中加上rephrase
#TODO: 过滤port数据集
i = 0
edit_needed = 0
data_len = len(data)
#print(data_len)
while i < len(data):
    
    #if data[i]["id"] not in [0,8,22,24,40,73]:
    
    if data[i]["id"] not in [0,1,2,3,5,8]:
        i = i + 1
        continue
    
    
    
    if "port" in data_path:
        if i >= len(data) - 6:
            break
    elif "multi" in data_path:
        if i >= len(data) - 60:
            break
    elif "sequential" in data_path:
        if data[i]["id"] >= 296:
            break
    output_datum = dict()
    if "port" in data_path:
        
        assert data[i]["type"] == "hop1" \
            and data[i+1]["type"] == "hop2" \
            and data[i+2]["type"] == "reasoning"
        #print(i)
        hop1_prompt = data[i]["text"]
        #hop1_prompt = port_template.format(hop1_prompt)
        hop2_prompt = data[i+1]["text"]
        #hop2_prompt = port_template.format(hop2_prompt)
        reasoning_prompt = data[i+2]["text"]
        reasoning_prompt_search = data[i+2]["text"]
        #reasoning_prompt = port_template.format(reasoning_prompt)
        hop1_subject = data[i]["concept"]
        hop2_subject = data[i+1]["concept"]
        hop1_relation = data[i]["relation"]
        hop2_relation = data[i+1]["relation"]
        implicit_answer = data[i]["labels"]
        explicit_answer = data[i+2]["labels"]
        #print(explicit_answer)
        i = i + 3

        hop1_last_tok_id = get_tgt_tok_id(hop1_prompt)
        hop1_check_tok_ids = [hop1_last_tok_id]
        hop2_last_tok_id = get_tgt_tok_id(hop2_prompt)
        hop2_check_tok_ids = [hop2_last_tok_id]
        reasoning_last_tok_id = get_tgt_tok_id(reasoning_prompt)
        reasoning_check_tok_ids = [reasoning_last_tok_id]
        implicit_toks = enc_tok(implicit_answer, avg=True)
        explicit_toks = enc_tok(explicit_answer, avg=True)

        calculate_hidden_generation_flow_port(editor, mt, hop1_prompt, hop2_prompt, hop1_subject, hop2_subject, hop1_relation, hop2_relation, reasoning_prompt, reasoning_prompt_search, hop1_check_tok_ids, hop2_check_tok_ids\
                                              , reasoning_check_tok_ids, [implicit_answer], [explicit_answer], [explicit_answer], [implicit_toks], [explicit_toks], [explicit_toks])
        editor.model.to("cpu")
        mt.model.to("cpu")
        editor.model = copy.deepcopy(origin_model)
        mt.model = editor.model
        '''
        port_type = "hop2"
        calculate_hidden_flow(mt, hop2_prompt, hop2_check_tok_ids, implicit_toks, explicit_toks)
        port_type = "reasoning"
        calculate_hidden_flow(mt, reasoning_prompt, reasoning_check_tok_ids, implicit_toks, explicit_toks)
        '''
        json.dump(multi_results, open(write_path, "w"), indent=4)
    elif "multi" in data_path:
        multi_prompt = data[i]["text"]
        multi_prompt_test = multi_template.format(multi_prompt)
        multi_prompt_rephrase = data[i]["rephrase"]
        multi_prompt_test_rephrase = multi_template_rephrase.format(multi_prompt_rephrase[0:-1])
        multi_subject = data[i]["concept"]
        multi_relation = data[i]["relation"]
        answer_list = data[i]["answer_success"]
        answer_list_new = data[i]["answer_miss"]

        i = i + 1

        last_tok_id = get_tgt_tok_id(multi_prompt)
        check_tok_ids = [last_tok_id]
        test_last_tok_id = get_tgt_tok_id(multi_prompt_test)
        test_check_tok_ids = [test_last_tok_id]

        answer_toks_list = list()
        for answer in answer_list:
            toks = enc_tok(answer, avg=True)
            answer_toks_list.append(toks)
            #multi_results["answer_dict"][answer] = toks 

        answer_toks_list_new = list()
        for answer_new in answer_list_new:
            toks_new = enc_tok(answer_new, avg=True)
            answer_toks_list_new.append(toks_new)  

           
        
        #print(multi_prompt)
        calculate_hidden_generation_flow(editor, mt, multi_prompt, multi_prompt_test, multi_prompt_test_rephrase, multi_subject, multi_relation, answer_list_new, answer_toks_list_new, check_tok_ids, test_check_tok_ids, answer_toks_list, answer_list)
        #calculate_hidden_generation_flow(editor, mt, multi_prompt, multi_subject, None, None, check_tok_ids, answer_toks_list, answer_list)
        #editor.model.to("cpu")
        #mt.model.to("cpu")
        #editor.model = copy.deepcopy(origin_model)
        mt.model = editor.model
        json.dump(multi_results, open(write_path, "w"), indent=4)
    elif "dc" in data_path:
        if data[i]["type"] == "complete" or data[i+1]["type"] == "complete":
            i = i + 1
            continue
        seq1_prompt = data[i]["text"][0:-1]
        seq2_prompt = data[i+1]["text"][0:-1]
        #sequential_prompt = seq1_prompt[0:-1] + " and " + seq2_prompt[0:-1]
        sequential_prompt = sequential_template.format(seq1_prompt, seq2_prompt)
        sequential_prompt_search = sequential_template_search.format(seq1_prompt, seq2_prompt)
        #seq1_prompt = sequential_template.format(seq1_prompt)
        #seq2_prompt = sequential_template.format(seq2_prompt)
        seq1_subject = data[i]["concept"]
        seq2_subject = data[i+1]["concept"]
        seq1_relation = data[i]["relation"]
        seq2_relation = data[i+1]["relation"]
        seq1_answer = data[i]["labels"]
        seq2_answer = data[i+1]["labels"]
        sequential_answer = seq1_answer + " and " + seq2_answer

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

        calculate_hidden_generation_flow_port(editor, mt, seq1_prompt, seq2_prompt, seq1_subject, seq2_subject, seq1_relation, seq2_relation, sequential_prompt, sequential_prompt_search, seq1_check_tok_ids, seq2_check_tok_ids\
                                              , sequential_check_tok_ids, [seq1_answer], [seq2_answer], [sequential_answer], [seq1_toks], [seq2_toks], [sequential_toks])
        editor.model.to("cpu")
        mt.model.to("cpu")
        editor.model = copy.deepcopy(origin_model)
        mt.model = editor.model
        json.dump(multi_results, open(write_path, "w"), indent=4)

if "port" in data_path:
    json.dump(multi_results, open(write_path, "w"), indent=4)
else:
    json.dump(multi_results, open(write_path, "w"), indent=4)







