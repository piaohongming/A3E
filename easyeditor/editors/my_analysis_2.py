import os, re, json, sys
# os.chdir("/root/autodl-tmp/zhaoyi/knowledge_locate/rome")
sys.path.append("../..") 
import torch, numpy
from collections import defaultdict
from easyeditor.util import nethook
from easyeditor.editors import BaseEditor
from easyeditor.models import A3EHyperParams

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
        plt.savefig("/home/hmpiao/rome/experiments/picture/tpatchertwo_attention_{}_l{}_h{}_t{}.png".format(state, layer, i, gen))
        plt.clf()
        plt.close()
    #fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)
    #pio.write_image(fig, './probe.jpg')


random.seed(0)
#torch.set_grad_enabled(False)

last_token_representation = None
last_token_representation_resubmit = None
last_token_representation_gen = None

t_global = None
t_global_change = None
port_type = ""
port_dict = {"explicit":{"prob":[], "rank":[]}, "implicit":{"prob":[], "rank":[]}}
port_results = dict()
port_results["hop2"] = copy.deepcopy(port_dict)
port_results["reasoning"] = copy.deepcopy(port_dict)

multi_results = dict()
multi_results["pre"] = dict()
multi_results["post"] = dict()
multi_results["pre"]["prob"] = dict()
multi_results["pre"]["rank"] = dict()
multi_results["post"]["prob"] = dict()
multi_results["post"]["rank"] = dict()
multi_results["pre"]["generate"] = list()
multi_results["post"]["generate"] = list()
#multi_results["answer_dict"] = dict()

port_template = "Q: The name of the anthem of Japan is? A: Kimigayo\n\
    Q: The name of the head of government of the place of birth of Lolita Robertson is? A: London Breed\n\
        Q: The place of birth of Mikuru Suzuki is? A: Japan\n\
            Q: The name of the continent which the place of birth of Lolita Robertson is part of is? A: North America\n\
                Q: {}?"
multi_template = "Q: Tim Dorsey, who has written the? A: Cadillac Beach, Nuclear Jellyfish, Triggerfish Twist, Hammerhead Ranch Motel, The Big Bamboo, Orange Crush (novel), Hurricane Punch, The Stingray Shuffle, Atomic Lobster, Torpedo Juice (novel), Florida Roadkill\n\
    Q: Jerusalem, which is the partner town of? A: NYC, New York, Praha, New York City, United States, Rio de Janeiro, NY, New York, NY, Prague, New York City, Tehran, Buenos Aires, Moscow, Manhattan\n\
        Q: Pushkin is the author of? A: The Fountain of Bakhchisaray, Eugene Onegin, The Tale of the Fisherman and the Fish, Poltava (poem), The Tale of the Golden Cockerel, Dubrovsky (novel), The Belkin Tales, Onegin, The Stone Guest (play), The Bronze Horseman (poem), The Queen of Spades (story), The Tale of the Priest and of His Workman Balda, The Gypsies, The Blizzard, Tatiana Larina, The Tale of the Dead Princess and the Seven Knights\n\
            Q: WWE is the owner of? A: WWE Classics on Demand, FCW Florida Heavyweight Championship, WWE Studios, WWE Films, WWE Network, NXT, FCW, WWE Classics On Demand, FCW Southern Heavyweight Championship, WCW, World Championship Wrestling, WCW, Inc., Florida Championship Wrestling, NXT Wrestling, Universal Wrestling Corporation, WWE NXT\n\
                Q: {}? A: "
sequential_template = "Q: The name of the head of government of San Francisco is and the name of the continent which San Francisco is? A: London Breed and North America\n\
    Q: The name of the continent which New York City is and the official language of the New York City is? A: North America and English\n\
        Q: The name of the head of government of New York City is and the name of the capital city of Japan is? A: Eric Adams and Tokyo\n\
            Q: The official language of Japan is and the name of the anthem of Japan is? A: Japanese and Kimigayo\n\
                Q: {}?"

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
    states_to_patch,  # A list of (token index, layername) triples to restore
    answer_toks_list, 
    answer_list,
    prompt,
    state
    ):

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)


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
        global last_token_representation
        global last_token_representation_resubmit
        if layer in save_layer_list:
            print("CNMCNMCNMCNMCNMCNMCNM")
            if last_token_representation_resubmit is None:
                last_token_representation_resubmit = x[0][-1]
            return x

        if layer not in resubmit_layer_list:
            return x
        if last_token_representation is not None:
            print(x[0].shape)
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
        '''
        tgt_toks = [tok_enc_1, tok_enc_2, ...]
        '''
        if layer not in patch_spec:
            return x
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

        for answer in answer_toks_list:

            prob = 0
            for tok in answer:
                prob += logits[tok]
            prob /= len(answer)
            rank = get_rank(logits, answer)

            answer_string = answer_list[answer_toks_list.index(answer)]
            if answer_string not in multi_results[state]["prob"].keys():
                multi_results[state]["prob"][answer_string] = dict()
                multi_results[state]["rank"][answer_string] = dict()

            if str(t_global_temp) in multi_results[state]["prob"][answer_string].keys():
                multi_results[state]["prob"][answer_string][str(t_global_temp)].append(prob)
                multi_results[state]["rank"][answer_string][str(t_global_temp)].append(rank)
            else:
                multi_results[state]["prob"][answer_string][str(t_global_temp)] = []
                multi_results[state]["rank"][answer_string][str(t_global_temp)] = []
                multi_results[state]["prob"][answer_string][str(t_global_temp)].append(prob)
                multi_results[state]["rank"][answer_string][str(t_global_temp)].append(rank)

        return x

    # With the patching rules defined, run the patched model in inference.
    
    outputs_exp_temp = model.generate(
        input_ids=inp['input_ids'],
        attention_mask=inp['attention_mask'],
        #min_new_tokens=len(target_new_tokens),
        max_new_tokens=18,
        #max_new_tokens=100, 
        do_sample=False,
        temperature=None,
        #num_beams=3,
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
    position_list = list()
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
            position_list.append(position)
        
    if "pre" == state:
        global edit_needed
        global data_len
        if len(answer_list_new) < len(answer_list):
            edit_needed = edit_needed + 1
            print("edit needed {}/{}".format(edit_needed, data_len))
        answer_toks_list = answer_toks_list_new
        answer_list = answer_list_new
    else:
        answer_toks_list[0] = answer_toks_list_new[0]
        answer_list[0] = answer_list_new[0]

    #print(list(patch_spec.keys()))
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=18,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        global t_global
        global t_global_change
        t_global = None
        t_global_change = None

    save_layer_list = ["model.norm"]
    with torch.no_grad(), nethook.TraceDict(
        model,
        save_layer_list,
        edit_output=patch_save,
    ) as td:
        outputs_for_save = model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=18,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]

    
    resubmit_layer_list = ["model.embed_tokens"]
    #resubmit_layer_list = ["model.norm"]
    with torch.no_grad(), nethook.TraceDict(
        origin_model,
        save_layer_list + resubmit_layer_list,
        edit_output=patch_resubmit,
    ) as td:
        outputs_for_resubmit = origin_model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=18,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
        
    resubmit_layer_list = ["model.embed_tokens"]
    #resubmit_layer_list = ["model.norm"]
    with torch.no_grad(), nethook.TraceDict(
        origin_model,
        resubmit_layer_list,
        edit_output=patch_resubmit_resubmit,
    ) as td:
        outputs_for_resubmit_resubmit = origin_model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=18,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None,
                #num_beams=3,
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]

    
    resubmit_layer_list = ["model.embed_tokens"]
    #resubmit_layer_list = ["model.norm"]
    with torch.no_grad(), nethook.TraceDict(
        model,
        save_layer_list + resubmit_layer_list,
        edit_output=patch_resubmit_gen,
    ) as td:
        outputs_for_resubmit_gen = model.generate(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                #min_new_tokens=len(target_new_tokens),
                max_new_tokens=18,
                #max_new_tokens=100, 
                do_sample=False,
                temperature=None
            ) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
    
    
    '''
    model_for_attentionmap = transformer_lens.HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B", move_to_device=False, hf_model=model, torch_dtype=torch.float32, device_map="balance", max_memory={0: "0GiB", 1: "31GiB", 2: "0GiB", 3: "0GiB", 4: "31GiB", 5: "0GiB", 6: "0GiB", 7: "31GiB"})
    outputs_for_attentionmap, cache_for_attentionmap = model_for_attentionmap.generate_with_cache(
        prompt_tokens=inp['input_ids'],
        #attention_mask=inp['attention_mask'],
        #min_new_tokens=len(target_new_tokens),
        max_gen_len=8,
        #max_new_tokens=100, 
        #do_sample=False,
        #temperature=None
        #tokenizer=mt.tokenizer
    )
    '''
    #out_test = model.forward(inp['input_ids'], output_attentions=True)
    #print(len(out_test.attentions))
    #print(out_test.attentions[0].shape)
    outputs_for_attentionmap, cache_for_attentionmap = generate_with_cache(
        model,
        prompt_tokens=inp['input_ids'],
        #attention_mask=inp['attention_mask'],
        #min_new_tokens=len(target_new_tokens),
        max_gen_len=18,
        #max_new_tokens=100, 
        #do_sample=False,
        #temperature=None
        #tokenizer=mt.tokenizer
    )
        
    assert outputs_exp.shape[0] == 1 # exp, compare
    # We report softmax probabilities for the answers_t token predictions of interest.
    return outputs_exp, answer_toks_list, answer_list, position_list, outputs_for_attentionmap, cache_for_attentionmap, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen

def generate_with_cache(
    model,
    prompt_tokens,
    max_gen_len,
):
    with torch.no_grad():
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
            
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            #TODO: logits = forward(tokens, prev_pos)
            out = model.forward(tokens, prev_pos, output_attentions=True)
            cache_list.append(out.attentions)
            
        stop_tokens = torch.tensor(list([mt.tokenizer.eos_token_id]), device="cuda")

        for cur_pos in range(min_prompt_len, total_len):
            #TODO: logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            out = model.forward(tokens[:, prev_pos:cur_pos], past_key_values=kvcache, output_attentions=True)
            #print(out)
            kvcache = out.past_key_values
            cache_list.append(out.attentions)

            next_token = torch.argmax(out.logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
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

def debias_states_list(editor, model, num_layers, inp, check_tok_ids, answer_toks_list, answer_list, prompt, subject, target_new, target_new_tok):
    #global editor
    '''
    for tnum in check_tok_ids:
        for layer in range(0, num_layers):
            outputs_exp, answer_toks_list, answer_list, position_list = trace_with_patch_list(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_toks_list,
                answer_list,
                prompt,
                state="pre"
            )
    '''
    #model_copy = copy.deepcopy(model)
    
    states_to_patch = []
    for tnum in check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(model, layer)))
    outputs_exp, answer_toks_list, answer_list, position_list, outputs_for_attentionmap, cache_for_attentionmap, outputs_for_save, outputs_for_resubmit, outputs_for_resubmit_resubmit, outputs_for_resubmit_gen = trace_with_patch_list(
        model,
        inp,
        states_to_patch,
        answer_toks_list,
        answer_list,
        prompt,
        state="pre"
    )
    #sorted_position_list = sorted(position_list)
    #target_new = answer_list[position_list.index(sorted_position_list[0])]

    
    #print("phmphmphm")
    if isinstance(target_new, list):
        edited_model = editor.simple_edit(
            prompts=[prompt],
            target_new=[target_new[0]],
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

        edited_model = editor.simple_edit(
            prompts=[prompt],
            target_new=[target_new[1]],
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

        answer_list_post = [target_new[0], target_new[1]]
        answer_list_post.extend(answer_list)
        answer_toks_list_post = [target_new_tok[0], target_new_tok[1]]
        answer_toks_list_post.extend(answer_toks_list)
    else:
        edited_model = editor.simple_edit(
            prompts=[prompt],
            target_new=[target_new],
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

        answer_list_post = [target_new]
        answer_list_post.extend(answer_list)
        answer_toks_list_post = [target_new_tok]
        answer_toks_list_post.extend(answer_toks_list)

    for key in edited_model.model.model.state_dict().keys():
        print(key)

    states_to_patch = []
    for tnum in check_tok_ids:
        for layer in range(0, num_layers):
            states_to_patch.append((tnum, layername(edited_model.model.model, layer)))
    outputs_exp_post, answer_toks_list_post, answer_list_post, position_list_post, outputs_for_attentionmap_post, cache_for_attentionmap_post, outputs_for_save_post, outputs_for_resubmit_post, outputs_for_resubmit_resubmit_post, outputs_for_resubmit_gen_post = trace_with_patch_list(
        #edited_model.model,
        edited_model.model.model,
        inp,
        states_to_patch,
        answer_toks_list_post,
        answer_list_post,
        prompt,
        state="post"
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

    gen_text_check = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_attentionmap.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_check = ''
    char_tok_dict_check = dict()
    j = 0
    for t in range(len(gen_text_check)):
        
        t_1 = gen_text_check[t]
        for i in range(len(t_1)):
            gen_text_1_check = gen_text_1_check + t_1[i]
            char_tok_dict_check[j] = t
            j = j + 1

    gen_text_check_post = [mt.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_for_attentionmap_post.detach().cpu().numpy().tolist()[0]]
    #gen_text_1 = ''.join(gen_text).strip()
    #增加每个字符串对应哪个token
    gen_text_1_check_post = ''
    char_tok_dict_check_post = dict()
    j = 0
    for t in range(len(gen_text_check_post)):
        
        t_1 = gen_text_check_post[t]
        for i in range(len(t_1)):
            gen_text_1_check_post = gen_text_1_check_post + t_1[i]
            char_tok_dict_check_post[j] = t
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

    
    multi_results["pre"]["generate"].append(
        {
            "prompt": prompt,
            "gen_tok": outputs_exp.detach().cpu().numpy().tolist()[0],
            "gen_tok_check": outputs_for_attentionmap.detach().cpu().numpy().tolist()[0],
            "gen_text": gen_text_1,
            "gen_text_check": gen_text_1_check, 
            "gen_text_save": gen_text_1_save,
            "gen_text_resubmit": gen_text_1_resubmit,
            "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit,
            "gen_text_resubmit_gen": gen_text_1_resubmit_gen,
            "answer_list": answer_list,
            "answer_toks_list": answer_toks_list,
            "char_tok_dict": char_tok_dict,
        }
    )
    
    multi_results["post"]["generate"].append(
        {
            "prompt": prompt,
            "gen_tok": outputs_exp_post.detach().cpu().numpy().tolist()[0],
            "gen_tok_check": outputs_for_attentionmap_post.detach().cpu().numpy().tolist()[0],
            "gen_text": gen_text_1_post,
            "gen_text_check": gen_text_1_check_post, 
            "gen_text_save": gen_text_1_save_post,
            "gen_text_resubmit": gen_text_1_resubmit_post,
            "gen_text_resubmit_resubmit": gen_text_1_resubmit_resubmit_post,
            "gen_text_resubmit_gen": gen_text_1_resubmit_gen_post,
            "answer_list": answer_list_post,
            "answer_toks_list": answer_toks_list_post,
            "char_tok_dict": char_tok_dict_post
        }
    )
    
    for i in range(len(cache_for_attentionmap)):
        if i <= 5:
            cache = cache_for_attentionmap[i]
            cache_post = cache_for_attentionmap_post[i]
            attention_pattern = cache[24][0]
            attention_pattern_post = cache_post[24][0]
            print("the shape of attention is: {}".format(attention_pattern.shape))
            #imshow(attention_pattern, layer=24, state="pre", gen=i)
            #imshow(attention_pattern_post, layer=24, state="post", gen=i)
    
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
    editor, mt, prompt, subject, target_new, target_new_tok, check_tok_ids, answer_toks_list, answer_list
    ):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt])

    debias_states_list(editor, mt.model, mt.num_layers, inp, check_tok_ids, answer_toks_list, answer_list, prompt, subject, target_new, target_new_tok)
    
    return 
'''
model_name = "/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B"
torch_dtype = torch.float32
device_map = 'balanced'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map, max_memory={0: "0GiB", 1: "20GiB", 2: "0GiB", 3: "0GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "31GiB"})
#{0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "31GiB", 7: "0GiB"}
#{0: "20GiB", 1: "31GiB", 2: "0GiB", 3: "31GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
'''

hparams = A3EHyperParams.from_hparams('/home/hmpiao/EasyEdit_method/hparams/A3E/llama3-8b.yaml')
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
write_path = "/data2/hmpiao/FME/log_analysis/multinormtokedit_tpatchertwo_attention_2.json"
#write_path = "/data2/hmpiao/FME/log_analysis/multinormtokedit_grace_attention.json"
#write_path = "/data2/hmpiao/FME/log_analysis/sequentialnormtokedit.json"

#data_path = "/home/hmpiao/EasyEdit/data/demo_port_analysis.json"
data_path = "/home/hmpiao/EasyEdit/data/demo_multi_analysis.json"
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
W = mt.model.state_dict()['lm_head.weight']
Norm_W = mt.model.state_dict()['model.norm.weight']
Norm = mt.model.model.norm
origin_model = copy.deepcopy(mt.model)

output_data = list()

#TODO: 在数据集中加上rephrase
#TODO: 过滤port数据集
i = 0
edit_needed = 0
data_len = len(data)
#print(data_len)
while i < len(data):
    if data[i]["id"] != 0:
        i = i + 1
        continue
    '''
    if i >= 9:
        break
    '''
    if "port" in data_path:
        if i >= len(data) - 6:
            break
    elif "multi" in data_path:
        if i >= len(data) - 60:
            break
    elif "sequential" in data_path:
        if i + 4 >= len(data) - 12:
            break
    output_datum = dict()
    if "port" in data_path:
        
        assert data[i]["type"] == "hop1" \
            and data[i+1]["type"] == "hop2" \
            and data[i+2]["type"] == "reasoning"
        #print(i)
        hop2_prompt = data[i+1]["text"]
        hop2_prompt = port_template.format(hop2_prompt)
        reasoning_prompt = data[i+2]["text"]
        reasoning_prompt = port_template.format(reasoning_prompt)
        implicit_answer = data[i]["labels"]
        explicit_answer = data[i+2]["labels"]
        i = i + 3

        hop2_last_tok_id = get_tgt_tok_id(hop2_prompt)
        hop2_check_tok_ids = [hop2_last_tok_id]
        reasoning_last_tok_id = get_tgt_tok_id(reasoning_prompt)
        reasoning_check_tok_ids = [reasoning_last_tok_id]
        implicit_toks = enc_tok(implicit_answer, avg=True)
        explicit_toks = enc_tok(explicit_answer, avg=True)

        port_type = "hop2"
        calculate_hidden_flow(mt, hop2_prompt, hop2_check_tok_ids, implicit_toks, explicit_toks)
        port_type = "reasoning"
        calculate_hidden_flow(mt, reasoning_prompt, reasoning_check_tok_ids, implicit_toks, explicit_toks)
        json.dump(port_results, open(write_path, "w"), indent=4)
    elif "multi" in data_path:
        answer_list = list()
        answer_list_new = list()
        
        while data[i]["type"] == "incomplete":
            answer_list.append(data[i]["labels"])
            if "new_labels" in data[i].keys():
                answer_list_new.append(data[i]["new_labels"])
            i = i + 1
        assert data[i]["type"] == "complete"
        multi_prompt = data[i]["text"]
        multi_prompt = multi_template.format(multi_prompt)
        multi_subject = data[i]["concept"]

        i = i + 1

        last_tok_id = get_tgt_tok_id(multi_prompt)
        check_tok_ids = [last_tok_id]

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
        calculate_hidden_generation_flow(editor, mt, multi_prompt, multi_subject, answer_list_new[0:2], answer_toks_list_new[0:2], check_tok_ids, answer_toks_list, answer_list)
        #calculate_hidden_generation_flow(editor, mt, multi_prompt, multi_subject, None, None, check_tok_ids, answer_toks_list, answer_list)
        json.dump(multi_results, open(write_path, "w"), indent=4)
    elif "sequential" in data_path:
        answer_1 = data[i+1]["labels"]

        if "new_labels" in data[i+1].keys():
            answer_1_new = data[i+1]["new_labels"]

        answer_2 = data[i+4]["labels"]

        if "new_labels" in data[i+4].keys():
            answer_2_new = data[i+4]["new_labels"]

        sequential_subject = data[i+1]["concept"]
        prompt_1 = data[i+1]["text"]
        prompt_2 = data[i+4]["text"]
        sequential_prompt = prompt_1 + " and " + prompt_2
        sequential_prompt = sequential_template.format(sequential_prompt)
        i = i + 3

        last_tok_id = get_tgt_tok_id(sequential_prompt)
        check_tok_ids = [last_tok_id]

        answer_toks_list = list()
        toks_1 = enc_tok(answer_1, avg=True)
        toks_1_new = enc_tok(answer_1_new, avg=True)

        toks_2 = enc_tok(answer_2, avg=True)
        toks_2_new = enc_tok(answer_2_new, avg=True)
        answer_toks_list.append(toks_1)
        answer_toks_list.append(toks_2)
        #multi_results["answer_dict"][answer_1] = toks_1
        #multi_results["answer_dict"][answer_2] = toks_2

        calculate_hidden_generation_flow(editor, mt, sequential_prompt, sequential_subject, answer_1_new, toks_1_new, check_tok_ids, answer_toks_list, [answer_1, answer_2])
        json.dump(multi_results, open(write_path, "w"), indent=4)
if "port" in data_path:
    json.dump(port_results, open(write_path, "w"), indent=4)
else:
    json.dump(multi_results, open(write_path, "w"), indent=4)







