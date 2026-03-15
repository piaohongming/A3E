import matplotlib.pyplot as plt
import json
import numpy as np
import os
import copy
from rouge_score import rouge_scorer, scoring
import scipy
#import nltk
#nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

MISS_KEY = 0
CORRECT_0 = list()
CORRECT_1 = list()
CORRECT_2 = list()
CORRECT_FIRST = list()

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

def plot(x, ys, title, save_path=None):
    colors = ['steelblue','darkred','darkgreen', 'gold', 'orchid', "grey", "red", "lime"]
    plt.figure()
    for i in range(len(ys)):
        plt.plot(x, ys[i], linewidth = 0.7, marker='D', mec='black', color=colors[i])

    plt.title(title)
    
    #plt.legend(prop = { "size": 9 })
    # 添加横向网格线
    plt.grid()
    # 显示图表
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def multi_plot(data):
    global MISS_KEY
    #处理pre
    data_pre = data["pre"]
    answer_pre_success_0_mean = list()
    answer_pre_success_0_max = list()
    answer_pre_success_0_min = list()
    answer_pre_miss_0_mean = list()
    
    #print(data["pre"]["origin"]["rank"]["Syria"]["329"][0:32])
    #print(data["post"]["origin"]["rank"]["Syria"]["329"][0:32])

    for gen_data in data_pre["generate"]:
        prompt = gen_data["prompt"]
        gen_text = gen_data["gen_text"]
        gen_text_resubmit = gen_data["gen_text_resubmit"]
        answer_list_success = gen_data["answer_list_success"]
        answer_list_miss = gen_data["answer_list_miss"]
        insert_tok = gen_data["insert_tok"]
        answer_pre_success_0_temp = list()
        answer_pre_miss_0_temp = list()
        for answer in answer_list_success:
            if answer not in data_pre["origin"]["rank"].keys():
                MISS_KEY += 1
                continue
            try:
                assert len(data_pre["origin"]["rank"][answer][insert_tok]) % 32 == 0
            except:
                MISS_KEY += 1
                continue
            answer_pre_success_0_temp.extend(data_pre["origin"]["rank"][answer][insert_tok][0:32])
            data_pre["origin"]["rank"][answer][insert_tok] = data_pre["origin"]["rank"][answer][insert_tok][32:]
        for answer in answer_list_miss:
            '''
            if answer not in data_pre["origin"]["rank"].keys():
                MISS_KEY += 1
                continue
            '''
            assert len(data_pre["origin"]["rank"][answer][insert_tok]) % 32 == 0
            answer_pre_miss_0_temp.extend(data_pre["origin"]["rank"][answer][insert_tok][0:32])
            data_pre["origin"]["rank"][answer][insert_tok] = data_pre["origin"]["rank"][answer][insert_tok][32:]
        
        
        for i in range(32):
            try:
                if len(answer_pre_success_0_temp) > 0: 
                    answer_pre_success_0_mean.append(np.mean(answer_pre_success_0_temp[i::32]))
                    answer_pre_success_0_max.append(np.max(answer_pre_success_0_temp[i::32]))
                    answer_pre_success_0_min.append(np.min(answer_pre_success_0_temp[i::32]))
                if len(answer_pre_miss_0_temp) > 0:
                    answer_pre_miss_0_mean.append(np.mean(answer_pre_miss_0_temp[i::32]))
            except:
                pass

    #print(answer_pre_success_0_mean)
    #print(answer_pre_success_0_max)
    
    answer_pre_success_0_mean = [np.mean(answer_pre_success_0_mean[i::32]) for i in range(32)]
    answer_pre_success_0_max = [np.mean(answer_pre_success_0_max[i::32]) for i in range(32)]
    answer_pre_success_0_min = [np.mean(answer_pre_success_0_min[i::32]) for i in range(32)]
    answer_pre_miss_0_mean = [np.mean(answer_pre_miss_0_mean[i::32]) for i in range(32)]
    plot(list(range(32)), [answer_pre_success_0_mean], title="answer_pre_success_0_mean", save_path="/data2/hmpiao/FME/logits_len_loss/answer_pre_success_0_mean.jpg")
    plot(list(range(32)), [answer_pre_success_0_max], title="answer_pre_success_0_max", save_path="/data2/hmpiao/FME/logits_len_loss/answer_pre_success_0_max.jpg")
    plot(list(range(32)), [answer_pre_success_0_min], title="answer_pre_success_0_min", save_path="/data2/hmpiao/FME/logits_len_loss/answer_pre_success_0_min.jpg")
    plot(list(range(32)), [answer_pre_miss_0_mean], title="answer_pre_miss_0_mean", save_path="/data2/hmpiao/FME/logits_len_loss/answer_pre_miss_0_mean.jpg")

    #处理全部post
    data_pre = copy.deepcopy(data["post"])
    answer_pre_success_0_mean = list()
    answer_pre_success_0_max = list()
    answer_pre_success_0_min = list()
    answer_pre_miss_0_mean = list()
    
    #print(data["pre"]["origin"]["rank"]["Syria"]["329"][0:32])
    #print(data["post"]["origin"]["rank"]["Syria"]["329"][0:32])

    for gen_data in data_pre["generate"]:
        prompt = gen_data["prompt"]
        gen_text = gen_data["gen_text"]
        gen_text_resubmit = gen_data["gen_text_resubmit"]
        answer_list_success = gen_data["answer_list_success"]
        answer_list_miss = gen_data["answer_list_miss"]
        insert_tok = gen_data["insert_tok"]
        answer_pre_success_0_temp = list()
        answer_pre_miss_0_temp = list()
        for answer in answer_list_success:
            if answer not in data_pre["origin"]["rank"].keys():
                MISS_KEY += 1
                continue
            try:
                assert len(data_pre["origin"]["rank"][answer][insert_tok]) % 32 == 0
            except:
                MISS_KEY += 1
                continue
            answer_pre_success_0_temp.extend(data_pre["origin"]["rank"][answer][insert_tok][0:32])
            data_pre["origin"]["rank"][answer][insert_tok] = data_pre["origin"]["rank"][answer][insert_tok][32:]
        for answer in answer_list_miss:
            '''
            if answer not in data_pre["origin"]["rank"].keys():
                MISS_KEY += 1
                continue
            '''
            assert len(data_pre["origin"]["rank"][answer][insert_tok]) % 32 == 0
            answer_pre_miss_0_temp.extend(data_pre["origin"]["rank"][answer][insert_tok][0:32])
            data_pre["origin"]["rank"][answer][insert_tok] = data_pre["origin"]["rank"][answer][insert_tok][32:]
        
        
        for i in range(32):
            try:
                if len(answer_pre_success_0_temp) > 0: 
                    answer_pre_success_0_mean.append(np.mean(answer_pre_success_0_temp[i::32]))
                    answer_pre_success_0_max.append(np.max(answer_pre_success_0_temp[i::32]))
                    answer_pre_success_0_min.append(np.min(answer_pre_success_0_temp[i::32]))
                if len(answer_pre_miss_0_temp) > 0:
                    answer_pre_miss_0_mean.append(np.mean(answer_pre_miss_0_temp[i::32]))
            except:
                pass

    #print(answer_pre_success_0_mean)
    #print(answer_pre_success_0_max)
    
    answer_pre_success_0_mean = [np.mean(answer_pre_success_0_mean[i::32]) for i in range(32)]
    answer_pre_success_0_max = [np.mean(answer_pre_success_0_max[i::32]) for i in range(32)]
    answer_pre_success_0_min = [np.mean(answer_pre_success_0_min[i::32]) for i in range(32)]
    answer_pre_miss_0_mean = [np.mean(answer_pre_miss_0_mean[i::32]) for i in range(32)]
    plot(list(range(32)), [answer_pre_success_0_mean], title="answer_post_success_0_mean", save_path="/data2/hmpiao/FME/logits_len_loss/answer_post_success_0_mean.jpg")
    plot(list(range(32)), [answer_pre_success_0_max], title="answer_post_success_0_max", save_path="/data2/hmpiao/FME/logits_len_loss/answer_post_success_0_max.jpg")
    plot(list(range(32)), [answer_pre_success_0_min], title="answer_post_success_0_min", save_path="/data2/hmpiao/FME/logits_len_loss/answer_post_success_0_min.jpg")
    plot(list(range(32)), [answer_pre_miss_0_mean], title="answer_post_miss_0_mean", save_path="/data2/hmpiao/FME/logits_len_loss/answer_post_miss_0_mean.jpg")   

    #处理post:1.分类
    data_post = copy.deepcopy(data["post"])
    data_post_origin_big_total = list()
    data_post_origin_big_one = list()
    data_post_origin_big_two = list()
    data_post_origin_small_total = list()
    data_post_origin_small_one = list()
    data_post_origin_small_two = list()
    data_post_hopping_big_total = list()
    data_post_hopping_big_one = list()
    data_post_hopping_big_another = list()
    data_post_hopping_big_two = list()
    data_post_hopping_small_total = list()
    data_post_hopping_small_one = list()
    data_post_hopping_small_another = list()
    data_post_hopping_small_two = list()
    
    for gen_data in data_post["generate"]:
        prompt = gen_data["prompt"]
        gen_text = gen_data["gen_text"]
        gen_text_resubmit = gen_data["gen_text_resubmit"]
        answer_list_success = gen_data["answer_list_success"]
        answer_list_miss = gen_data["answer_list_miss"]
        effect = 0
        for answer_success in answer_list_success:
            position, position_before = find_sublist_position(gen_text, answer_success, prompt)
            if position == 10000:
                effect = effect + 1
        if effect > 2:
            correct = list()
            correct_hopping = list()
            for answer_miss_index in range(len(answer_list_miss)):
                position, position_before = find_sublist_position(gen_text, answer_list_miss[answer_miss_index], prompt)
                if position != 10000:
                    correct.append(answer_miss_index)
                position, position_before = find_sublist_position(gen_text_resubmit, answer_list_miss[answer_miss_index], prompt)
                if position != 10000:
                    correct_hopping.append(answer_miss_index)
            if len(correct) == 0:
                data_post_origin_big_total.append(gen_data)
            if len(correct) == 1:
                data_post_origin_big_one.append(gen_data)
            if len(correct) == 2:
                data_post_origin_big_two.append(gen_data)
            if len(correct_hopping) == 0:
                data_post_hopping_big_total.append(gen_data)
            if len(correct_hopping) == 1 and correct_hopping[0] not in correct:
                data_post_hopping_big_another.append(gen_data)
            if len(correct_hopping) == 1 and correct_hopping[0] in correct:
                data_post_hopping_big_one.append(gen_data)
            if len(correct_hopping) == 2:
                data_post_hopping_big_two.append(gen_data)
            #print(f"big {len(correct)} {len(correct_hopping)}")
        else:
            correct = list()
            correct_hopping = list()
            for answer_miss_index in range(len(answer_list_miss)):
                position, position_before = find_sublist_position(gen_text, answer_list_miss[answer_miss_index], prompt)
                if position != 10000:
                    correct.append(answer_miss_index)
                position, position_before = find_sublist_position(gen_text_resubmit, answer_list_miss[answer_miss_index], prompt)
                if position != 10000:
                    correct_hopping.append(answer_miss_index)
            if len(correct) == 0:
                data_post_origin_small_total.append(gen_data)
            if len(correct) == 1:
                data_post_origin_small_one.append(gen_data)
            if len(correct) == 2:
                #print(gen_data["prompt"])
                #print(gen_data["gen_text"])
                #print(gen_data["answer_list_miss"])
                data_post_origin_small_two.append(gen_data)
            if len(correct_hopping) == 0:
                data_post_hopping_small_total.append(gen_data)
            if len(correct_hopping) == 1 and correct_hopping[0] not in correct:
                data_post_hopping_small_another.append(gen_data)
            if len(correct_hopping) == 1 and correct_hopping[0] in correct:
                data_post_hopping_small_one.append(gen_data)
            if len(correct_hopping) == 2:
                data_post_hopping_small_two.append(gen_data)
            #print(f"small {len(correct)} {len(correct_hopping)}")

    #print(data["post"]["origin"]["rank"]["Syria"]["329"][0:32])
    
    total_dict = {"data_post_origin_big_total": data_post_origin_big_total, "data_post_origin_small_total": data_post_origin_small_total, "data_post_hopping_big_total": data_post_hopping_big_total, "data_post_hopping_small_total": data_post_hopping_small_total}
    one_dict = {"data_post_origin_big_one": data_post_origin_big_one, "data_post_origin_small_one": data_post_origin_small_one, "data_post_hopping_big_one": data_post_hopping_big_one, "data_post_hopping_big_another": data_post_hopping_big_another, "data_post_hopping_small_one": data_post_hopping_small_one, "data_post_hopping_small_another": data_post_hopping_small_another}
    two_dict = {"data_post_origin_big_two": data_post_origin_big_two, "data_post_origin_small_two": data_post_origin_small_two, "data_post_hopping_big_two": data_post_hopping_big_two, "data_post_hopping_small_two": data_post_hopping_small_two}
    
    total_dict_len = {"data_post_origin_big_total": len(data_post_origin_big_total), "data_post_origin_small_total": len(data_post_origin_small_total), "data_post_hopping_big_total": len(data_post_hopping_big_total), "data_post_hopping_small_total": len(data_post_hopping_small_total)}
    one_dict_len = {"data_post_origin_big_one": len(data_post_origin_big_one), "data_post_origin_small_one": len(data_post_origin_small_one), "data_post_hopping_big_one": len(data_post_hopping_big_one), "data_post_hopping_big_another": len(data_post_hopping_big_another), "data_post_hopping_small_one": len(data_post_hopping_small_one), "data_post_hopping_small_another": len(data_post_hopping_small_another)}
    two_dict_len = {"data_post_origin_big_two": len(data_post_origin_big_two), "data_post_origin_small_two": len(data_post_origin_small_two), "data_post_hopping_big_two": len(data_post_hopping_big_two), "data_post_hopping_small_two": len(data_post_hopping_small_two)}
    print(total_dict_len)
    print(one_dict_len)
    print(two_dict_len)
    print(len(data_post["generate"]))
    '''
    for p in two_dict["data_post_origin_small_two"]: 
        print(p["gen_text"])
        print(p["answer_list_success"])
        print(p["answer_list_miss"])
    '''
    #print(data_post_origin_small_one)
    #处理post:2. 开始处理
    #print(one_dict["data_post_origin_big_one"][0]["answer_list_success"][0])
    #print(data["post"]["origin"]["rank"][one_dict["data_post_origin_big_one"][0]["answer_list_success"][0]][one_dict["data_post_origin_big_one"][0]["insert_tok"]][0:32])
    
    for key in total_dict.keys():

        data_post = copy.deepcopy(data["post"])
        answer_post_success_0_mean = list()
        answer_post_success_0_max = list()
        answer_post_success_0_min = list()
        answer_post_success_1_mean = list()
        answer_post_success_1_max = list()
        answer_post_success_1_min = list()
        answer_post_success_2_mean = list()
        answer_post_success_2_max = list()
        answer_post_success_2_min = list()
        answer_post_miss_0_mean = list()
        answer_post_miss_0 = list()
        answer_post_miss_1 = list()
        answer_post_miss_2 = list()
        answer_post_correct_0 = list()
        answer_post_correct_1 = list()
        answer_post_correct_2 = list()

        if "origin" in key:
            origin_or_hopping = "origin"
        elif "hopping" in key:
            origin_or_hopping = "hopping"
        data_temp = total_dict[key]
        for gen_data in data_temp:
            prompt = gen_data["prompt"]
            gen_text = gen_data["gen_text"]
            gen_text_resubmit = gen_data["gen_text_resubmit"]
            answer_list_success = gen_data["answer_list_success"]
            answer_list_miss = gen_data["answer_list_miss"]
            insert_tok = gen_data["insert_tok"]
            answer_post_success_0_temp = list()
            answer_post_miss_0_temp = list()
            for answer in answer_list_success:
                if answer not in data_post[origin_or_hopping]["rank"].keys():
                    MISS_KEY += 1
                    continue
                #print(data_post[origin_or_hopping]["rank"][answer].keys())
                assert len(data_post[origin_or_hopping]["rank"][answer][insert_tok]) % 32 == 0 
                answer_post_success_0_temp.extend(data_post[origin_or_hopping]["rank"][answer][insert_tok][0:32])
                data_post[origin_or_hopping]["rank"][answer][insert_tok] = data_post[origin_or_hopping]["rank"][answer][insert_tok][32:]
            for answer in answer_list_miss:
                '''
                if answer not in data_post[origin_or_hopping]["rank"].keys():
                    MISS_KEY += 1
                    continue
                '''
                assert len(data_post[origin_or_hopping]["rank"][answer][insert_tok]) % 32 == 0 
                answer_post_miss_0_temp.extend(data_post[origin_or_hopping]["rank"][answer][insert_tok][0:32])
                data_post[origin_or_hopping]["rank"][answer][insert_tok] = data_post[origin_or_hopping]["rank"][answer][insert_tok][32:]
            for i in range(32):
                try:
                    if len(answer_post_success_0_temp) > 0:
                        answer_post_success_0_mean.append(np.mean(answer_post_success_0_temp[i::32]))
                        answer_post_success_0_max.append(np.max(answer_post_success_0_temp[i::32]))
                        answer_post_success_0_min.append(np.min(answer_post_success_0_temp[i::32]))
                    if len(answer_post_miss_0_temp) > 0:   
                        answer_post_miss_0_mean.append(np.mean(answer_post_miss_0_temp[i::32]))
                except:
                    pass
    
        answer_post_success_0_mean = [np.mean(answer_post_success_0_mean[i::32]) for i in range(32)]
        answer_post_success_0_max = [np.mean(answer_post_success_0_max[i::32]) for i in range(32)]
        answer_post_success_0_min = [np.mean(answer_post_success_0_min[i::32]) for i in range(32)]
        answer_post_miss_0_mean = [np.mean(answer_post_miss_0_mean[i::32]) for i in range(32)]
        if not os.path.exists(f'/data2/hmpiao/FME/logits_len_loss/{key}/'):
            os.mkdir(f'/data2/hmpiao/FME/logits_len_loss/{key}/')

        plot(list(range(32)), [answer_post_success_0_mean], title="answer_post_success_0_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_mean.jpg")
        plot(list(range(32)), [answer_post_success_0_max], title="answer_post_success_0_max", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_max.jpg")
        plot(list(range(32)), [answer_post_success_0_min], title="answer_post_success_0_min", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_min.jpg")
        plot(list(range(32)), [answer_post_miss_0_mean], title="answer_post_miss_0_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_miss_0_mean.jpg")   
    
    
    
    for key in one_dict.keys():
        data_post = copy.deepcopy(data["post"])
        answer_post_success_0_mean = list()
        answer_post_success_0_max = list()
        answer_post_success_0_min = list()
        answer_post_success_1_mean = list()
        answer_post_success_1_max = list()
        answer_post_success_1_min = list()
        answer_post_success_2_mean = list()
        answer_post_success_2_max = list()
        answer_post_success_2_min = list()
        answer_post_miss_0_mean = list()
        answer_post_miss_0 = list()
        answer_post_miss_1 = list()
        answer_post_miss_2 = list()
        answer_post_correct_0 = list()
        answer_post_correct_1 = list()
        answer_post_correct_2 = list()

        data_temp = one_dict[key]
        for gen_data in data_temp:
            prompt = gen_data["prompt"]
            gen_text = gen_data["gen_text"]
            gen_text_resubmit = gen_data["gen_text_resubmit"]
            answer_list_success = gen_data["answer_list_success"]
            answer_list_miss = gen_data["answer_list_miss"]
            insert_tok = gen_data["insert_tok"]
            char_tok_dict = gen_data["char_tok_dict"]
            char_tok_dict_resubmit = gen_data["char_tok_dict_resubmit"]
            answer_miss_tok = None
            answer_correct = None
            answer_miss_ = None
            answer_post_success_0_temp = list()
            answer_post_success_1_temp = list()
            

            if "hopping" in key:
                for answer_miss in answer_list_miss:
                    position, position_before = find_sublist_position(gen_text_resubmit, answer_miss, prompt)
                    if position != 10000:
                        answer_correct = answer_miss
                        answer_correct_tok = str(char_tok_dict_resubmit[str(position)])
                    else:
                        answer_miss_ = answer_miss
            if "origin" in key:
                for answer_miss in answer_list_miss:
                    position, position_before = find_sublist_position(gen_text, answer_miss, prompt)
                    if position != 10000:
                        answer_correct = answer_miss
                        answer_correct_tok = str(char_tok_dict[str(position)])
                    else:
                        answer_miss_ = answer_miss

            for answer in answer_list_success:
                if answer not in data_post[origin_or_hopping]["rank"].keys():
                    MISS_KEY += 1
                    continue
                try:
                    assert len(data_post[origin_or_hopping]["rank"][answer][insert_tok]) % 32 == 0 
                    assert len(data_post[origin_or_hopping]["rank"][answer][answer_correct_tok]) % 32 == 0
                except:
                    MISS_KEY += 1
                    continue
                answer_post_success_0_temp.extend(data_post[origin_or_hopping]["rank"][answer][insert_tok][0:32])
                data_post[origin_or_hopping]["rank"][answer][insert_tok] = data_post[origin_or_hopping]["rank"][answer][insert_tok][32:]
                answer_post_success_1_temp.extend(data_post[origin_or_hopping]["rank"][answer][answer_correct_tok][0:32])
                data_post[origin_or_hopping]["rank"][answer][answer_correct_tok] = data_post[origin_or_hopping]["rank"][answer][answer_correct_tok][32:]

            
            for i in range(32):
                try:
                    if len(answer_post_success_0_temp) > 0:
                        answer_post_success_0_mean.append(np.mean(answer_post_success_0_temp[i::32]))
                        answer_post_success_0_max.append(np.max(answer_post_success_0_temp[i::32]))
                        answer_post_success_0_min.append(np.min(answer_post_success_0_temp[i::32]))
                    if len(answer_post_success_1_temp) > 0:
                        answer_post_success_1_mean.append(np.mean(answer_post_success_1_temp[i::32]))
                        answer_post_success_1_max.append(np.max(answer_post_success_1_temp[i::32]))
                        answer_post_success_1_min.append(np.min(answer_post_success_1_temp[i::32]))
                except:
                    pass


            #print(len(data_post[origin_or_hopping]["rank"][answer_correct][insert_tok]))
            answer_post_correct_0.extend(data_post[origin_or_hopping]["rank"][answer_correct][insert_tok])
            #print(len(answer_post_correct_0))
            answer_post_correct_1.extend(data_post[origin_or_hopping]["rank"][answer_correct][answer_correct_tok])
            answer_post_miss_0.extend(data_post[origin_or_hopping]["rank"][answer_miss_][insert_tok])
            answer_post_miss_1.extend(data_post[origin_or_hopping]["rank"][answer_miss_][answer_correct_tok])
    
        answer_post_success_0_mean = [np.mean(answer_post_success_0_mean[i::32]) for i in range(32)]
        answer_post_success_0_max = [np.mean(answer_post_success_0_max[i::32]) for i in range(32)]
        answer_post_success_0_min = [np.mean(answer_post_success_0_min[i::32]) for i in range(32)]
        answer_post_success_1_mean = [np.mean(answer_post_success_1_mean[i::32]) for i in range(32)]
        answer_post_success_1_max = [np.mean(answer_post_success_1_max[i::32]) for i in range(32)]
        answer_post_success_1_min = [np.mean(answer_post_success_1_min[i::32]) for i in range(32)]
        
        #print(len(answer_post_success_1_min))
        #print(len(answer_post_correct_0))

        answer_post_correct_0 = [np.mean(answer_post_correct_0[i::32]) for i in range(32)]
        answer_post_correct_1 = [np.mean(answer_post_correct_1[i::32]) for i in range(32)]
        answer_post_miss_0 = [np.mean(answer_post_miss_0[i::32]) for i in range(32)]
        answer_post_miss_1 = [np.mean(answer_post_miss_1[i::32]) for i in range(32)]
        
        
        if not os.path.exists(f'/data2/hmpiao/FME/logits_len_loss/{key}/'):
            os.mkdir(f'/data2/hmpiao/FME/logits_len_loss/{key}/')
        plot(list(range(32)), [answer_post_success_0_mean], title="answer_post_success_0_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_mean.jpg")
        plot(list(range(32)), [answer_post_success_0_max], title="answer_post_success_0_max", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_max.jpg")
        plot(list(range(32)), [answer_post_success_0_min], title="answer_post_success_0_min", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_min.jpg")
        plot(list(range(32)), [answer_post_success_1_mean], title="answer_post_success_1_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_1_mean.jpg")
        plot(list(range(32)), [answer_post_success_1_max], title="answer_post_success_1_max", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_1_max.jpg")
        plot(list(range(32)), [answer_post_success_1_min], title="answer_post_success_1_min", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_1_min.jpg")
        plot(list(range(32)), [answer_post_correct_0], title="answer_post_correct_0", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct_0.jpg")
        plot(list(range(32)), [answer_post_correct_1], title="answer_post_correct_1", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct_1.jpg")
        plot(list(range(32)), [answer_post_miss_0], title="answer_post_miss_0", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_miss_0.jpg")
        plot(list(range(32)), [answer_post_miss_1], title="answer_post_miss_1", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_miss_1.jpg")

    

    for key in two_dict.keys():
        data_post = copy.deepcopy(data["post"])
        answer_post_success_0_mean = list()
        answer_post_success_0_max = list()
        answer_post_success_0_min = list()
        answer_post_success_1_mean = list()
        answer_post_success_1_max = list()
        answer_post_success_1_min = list()
        answer_post_success_2_mean = list()
        answer_post_success_2_max = list()
        answer_post_success_2_min = list()
        answer_post_miss_0_mean = list()
        answer_post_correct0_0 = list()
        answer_post_correct0_1 = list()
        answer_post_correct0_2 = list()
        answer_post_correct1_0 = list()
        answer_post_correct1_1 = list()
        answer_post_correct1_2 = list()

        data_temp = two_dict[key]
        for gen_data in data_temp:
            prompt = gen_data["prompt"]
            gen_text = gen_data["gen_text"]
            gen_text_resubmit = gen_data["gen_text_resubmit"]
            answer_list_success = gen_data["answer_list_success"]
            answer_list_miss = gen_data["answer_list_miss"]
            insert_tok = gen_data["insert_tok"]
            char_tok_dict = gen_data["char_tok_dict"]
            char_tok_dict_resubmit = gen_data["char_tok_dict_resubmit"]
            answer_correct_tok_0 = None
            answer_correct_tok_1 = None
            answer_correct_0 = None
            answer_correct_1 = None
            answer_post_success_0_temp = list()
            answer_post_success_1_temp = list()
            answer_post_success_2_temp = list()

            if "hopping" in key:
                position_0, position_before_0 = find_sublist_position(gen_text_resubmit, answer_list_miss[0], prompt)
                assert position_0 != 10000
                answer_correct_tok_0 = str(char_tok_dict_resubmit[str(position_0)])
                position_1, position_before_1 = find_sublist_position(gen_text_resubmit, answer_list_miss[1], prompt)
                assert position_1 != 10000
                answer_correct_tok_1 = str(char_tok_dict_resubmit[str(position_1)])
                if position_0 < position_1:
                    answer_correct_0 = answer_list_miss[0]
                    answer_correct_1 = answer_list_miss[1]
                else:
                    answer_correct_0 = answer_list_miss[1]
                    answer_correct_1 = answer_list_miss[0]
                    temp = answer_correct_tok_0
                    answer_correct_tok_0 = answer_correct_tok_1
                    answer_correct_tok_1 = temp
            if "origin" in key:
                position_0, position_before_0 = find_sublist_position(gen_text, answer_list_miss[0], prompt)
                assert position_0 != 10000
                answer_correct_tok_0 = str(char_tok_dict[str(position_0)])
                position_1, position_before_1 = find_sublist_position(gen_text, answer_list_miss[1], prompt)
                assert position_1 != 10000
                answer_correct_tok_1 = str(char_tok_dict[str(position_1)])
                if position_0 < position_1:
                    answer_correct_0 = answer_list_miss[0]
                    answer_correct_1 = answer_list_miss[1]
                else:
                    answer_correct_0 = answer_list_miss[1]
                    answer_correct_1 = answer_list_miss[0]
                    temp = answer_correct_tok_0
                    answer_correct_tok_0 = answer_correct_tok_1
                    answer_correct_tok_1 = temp

            for answer in answer_list_success:
                if answer not in data_post[origin_or_hopping]["rank"].keys():
                    MISS_KEY += 1
                    continue
                try:
                    assert len(data_post[origin_or_hopping]["rank"][answer][insert_tok]) % 32 == 0 
                    assert len(data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_0]) % 32 == 0
                    assert len(data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_1]) % 32 == 0
                except:
                    MISS_KEY += 1
                    continue
                answer_post_success_0_temp.extend(data_post[origin_or_hopping]["rank"][answer][insert_tok][0:32])
                data_post[origin_or_hopping]["rank"][answer][insert_tok] = data_post[origin_or_hopping]["rank"][answer][insert_tok][32:]
                answer_post_success_1_temp.extend(data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_0][0:32])
                data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_0] = data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_0][32:]
                answer_post_success_2_temp.extend(data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_1][0:32])
                data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_1] = data_post[origin_or_hopping]["rank"][answer][answer_correct_tok_1][32:]
            
            for i in range(32):
                try:
                    if len(answer_post_success_0_temp) > 0:
                        answer_post_success_0_mean.append(np.mean(answer_post_success_0_temp[i::32]))
                        answer_post_success_0_max.append(np.max(answer_post_success_0_temp[i::32]))
                        answer_post_success_0_min.append(np.min(answer_post_success_0_temp[i::32]))
                    if len(answer_post_success_1_temp) > 0:    
                        answer_post_success_1_mean.append(np.mean(answer_post_success_1_temp[i::32]))
                        answer_post_success_1_max.append(np.max(answer_post_success_1_temp[i::32]))
                        answer_post_success_1_min.append(np.min(answer_post_success_1_temp[i::32]))
                    if len(answer_post_success_2_temp) > 0:    
                        answer_post_success_2_mean.append(np.mean(answer_post_success_2_temp[i::32]))
                        answer_post_success_2_max.append(np.max(answer_post_success_2_temp[i::32]))
                        answer_post_success_2_min.append(np.min(answer_post_success_2_temp[i::32]))
                except:
                    pass

            answer_post_correct0_0.extend(data_post[origin_or_hopping]["rank"][answer_correct_0][insert_tok])
            answer_post_correct0_1.extend(data_post[origin_or_hopping]["rank"][answer_correct_0][answer_correct_tok_0])
            answer_post_correct0_2.extend(data_post[origin_or_hopping]["rank"][answer_correct_0][answer_correct_tok_1])
            answer_post_correct1_0.extend(data_post[origin_or_hopping]["rank"][answer_correct_1][insert_tok])
            answer_post_correct1_1.extend(data_post[origin_or_hopping]["rank"][answer_correct_1][answer_correct_tok_0])
            answer_post_correct1_2.extend(data_post[origin_or_hopping]["rank"][answer_correct_1][answer_correct_tok_1])

        answer_post_success_0_mean = [np.mean(answer_post_success_0_mean[i::32]) for i in range(32)]
        answer_post_success_0_max = [np.mean(answer_post_success_0_max[i::32]) for i in range(32)]
        answer_post_success_0_min = [np.mean(answer_post_success_0_min[i::32]) for i in range(32)]
        answer_post_success_1_mean = [np.mean(answer_post_success_1_mean[i::32]) for i in range(32)]
        answer_post_success_1_max = [np.mean(answer_post_success_1_max[i::32]) for i in range(32)]
        answer_post_success_1_min = [np.mean(answer_post_success_1_min[i::32]) for i in range(32)]
        answer_post_success_2_mean = [np.mean(answer_post_success_2_mean[i::32]) for i in range(32)]
        answer_post_success_2_max = [np.mean(answer_post_success_2_max[i::32]) for i in range(32)]
        answer_post_success_2_min = [np.mean(answer_post_success_2_min[i::32]) for i in range(32)]
        answer_post_correct0_0 = [np.mean(answer_post_correct0_0[i::32]) for i in range(32)]
        answer_post_correct0_1 = [np.mean(answer_post_correct0_1[i::32]) for i in range(32)]
        answer_post_correct0_2 = [np.mean(answer_post_correct0_2[i::32]) for i in range(32)]
        answer_post_correct1_0 = [np.mean(answer_post_correct1_0[i::32]) for i in range(32)]
        answer_post_correct1_1 = [np.mean(answer_post_correct1_1[i::32]) for i in range(32)]
        answer_post_correct1_2 = [np.mean(answer_post_correct1_2[i::32]) for i in range(32)]
        
        if not os.path.exists(f'/data2/hmpiao/FME/logits_len_loss/{key}/'):
            os.mkdir(f'/data2/hmpiao/FME/logits_len_loss/{key}/')
        plot(list(range(32)), [answer_post_success_0_mean], title="answer_post_success_0_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_mean.jpg")
        plot(list(range(32)), [answer_post_success_0_max], title="answer_post_success_0_max", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_max.jpg")
        plot(list(range(32)), [answer_post_success_0_min], title="answer_post_success_0_min", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_0_min.jpg")
        plot(list(range(32)), [answer_post_success_1_mean], title="answer_post_success_1_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_1_mean.jpg")
        plot(list(range(32)), [answer_post_success_1_max], title="answer_post_success_1_max", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_1_max.jpg")
        plot(list(range(32)), [answer_post_success_1_min], title="answer_post_success_1_min", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_1_min.jpg")
        plot(list(range(32)), [answer_post_success_2_mean], title="answer_post_success_2_mean", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_2_mean.jpg")
        plot(list(range(32)), [answer_post_success_2_max], title="answer_post_success_2_max", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_2_max.jpg")
        plot(list(range(32)), [answer_post_success_2_min], title="answer_post_success_2_min", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_success_2_min.jpg")
        plot(list(range(32)), [answer_post_correct0_0], title="answer_post_correct0_0", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct0_0.jpg")
        plot(list(range(32)), [answer_post_correct0_1], title="answer_post_correct0_1", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct0_1.jpg")
        plot(list(range(32)), [answer_post_correct0_2], title="answer_post_correct0_2", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct0_2.jpg")
        plot(list(range(32)), [answer_post_correct1_0], title="answer_post_correct1_0", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct1_0.jpg")
        plot(list(range(32)), [answer_post_correct1_1], title="answer_post_correct1_1", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct1_1.jpg")
        plot(list(range(32)), [answer_post_correct1_2], title="answer_post_correct1_2", save_path=f"/data2/hmpiao/FME/logits_len_loss/{key}/answer_post_correct1_2.jpg")

    print(MISS_KEY)   

def multi_plot_easy(data):
    global CORRECT_0
    global CORRECT_1
    global CORRECT_2
    global CORRECT_FIRST

    

    all_correct = list()
    data_post = data["post"]
    length = len(data_post["generate"])
    #print(data["pre"]["origin"]["rank"]["Syria"]["329"][0:32])
    #print(data["post"]["origin"]["rank"]["Syria"]["329"][0:32])

    correct_0 = 0
    correct_1 = 0
    correct_2 = 0
    correct_first = 0
    rouge1_precision = 0
    rouge1_recall = 0
    rougel_precision = 0
    rougel_recall = 0
    #correct_3 = 0
    for gen_data in data_post["generate"]:
        
        prompt = gen_data["prompt"]
        gen_text = gen_data["gen_text"]
        answer_list_miss = gen_data["answer_list_miss"]
        #
        '''
        insert_tok = gen_data["insert_tok"]
        correct = 0
        for answer in answer_list_miss[0:2]:
            if answer in gen_text:
                correct = correct + 1
        
        if correct == 0:
            #print(data_post["generate"].index(gen_data))
            correct_0 += 1
        elif correct == 1:
            correct_1 += 1
            #print(data_post["generate"].index(gen_data))
        elif correct == 2:
            correct_2 += 1
            #print("mmm")
            position_1, position_before_1 = find_sublist_position(gen_text, answer_list_miss[0], prompt)
            position_2, position_before_2 = find_sublist_position(gen_text, answer_list_miss[1], prompt)
            if gen_text[min([position_1, position_2]):max(position_before_1, position_before_2)].count(",") <= 1 and position_1 < position_2:
                correct_first += 1
                #print("nnn")
            print(data_post["generate"].index(gen_data))
            all_correct.append(data_post["generate"].index(gen_data))
        '''
        inner_prefix = make_inputs(tok, [gen_text[len(prompt)+len("<|begin_of_text|>")+1:]])
        if inner_prefix["input_ids"].shape[1] <= 5:
            inner_toks_prefix = [tok.decode(inner_prefix["input_ids"][0][i]) for i in range(1,inner_prefix["input_ids"].shape[1])]
        else:
            inner_toks_prefix = [tok.decode(inner_prefix["input_ids"][0][i]) for i in range(1,6)]
        real_text = ""
        for t in range(len(inner_toks_prefix)):
        
            t_1 = inner_toks_prefix[t]
            for i in range(len(t_1)):
                real_text = real_text + t_1[i]
                
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=tok)
        #print(real_text)
        #print(data_locality[data_post["generate"].index(gen_data)]["locality_answer"])
        rougea1 = scorer.score(real_text, data_locality[data_post["generate"].index(gen_data)]["locality_answer"])
        #print(rougea1['rougeL'])
        #rougea2 = scorer.score(gen_text[len(prompt):], answer_list_miss[1] + " " + answer_list_miss[0])
        rouge1_precision += rougea1['rouge1'].fmeasure
        #rouge1_precision += rougea2['rouge1'].precision
        rouge1_recall += rougea1['rouge1'].recall
        #rouge1_recall += rougea2['rouge1'].recall
        rougel_precision += rougea1['rougeL'].fmeasure
        #print(rougea1['rougeL'].fmeasure)
        #rougel_precision += rougea2['rougeL'].precision
        rougel_recall += rougea1['rougeL'].recall
        #rougel_recall += rougea2['rougeL'].recall
        
        #elif correct == 3:
            #correct_3 += 1
    #print(length)
    #print(correct_0/length)
    #print(correct_1/length)
    #print(correct_2/length)
    #print(correct_0)
    #print(correct_1)
    CORRECT_0.append(correct_0/length)
    CORRECT_1.append(correct_1/length)
    CORRECT_2.append(correct_2/length)
    CORRECT_FIRST.append(correct_first/length)
    #print(rouge1_precision/(2*length))
    #print(rouge1_recall/(2*length))
    print("rouge-l")
    print(rougel_precision/(1*length))
    #print(rougel_recall/(2*length))
    return all_correct

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)
    

#data_path = "/data2/hmpiao/FME/log_analysis/port.json"
#data_path = "/data2/hmpiao/FME/log_analysis/multi_analysis_all_continue_onlytop10.json"
temporature = [0.02]#list(np.linspace(0.5, 2.5, 4))[2:3]
penalty = [5.0]#list(np.linspace(1.0, 5.0, 10))[8:9]
for tem in temporature:
    for pen in penalty:
        data_path_1 = "/home/hmpiao/hmpiao/log/t-patcher/sequential_dc_analysis_all_continue_alledits_again_time_local.json_0.02_10.0.json"
        data_path_2 = "/home/hmpiao/hmpiao/log/t-patcher/sequential_dc_analysis_all_continue_alledits_again_time_local.json_0.02_10.0.json"
        data_path_3 = "/home/hmpiao/hmpiao/log/t-patcher/sequential_dc_analysis_all_continue_alledits_again_time_local.json_0.02_10.0.json"
        
        data_path_locality = "/home/hmpiao/EasyEdit_method/data/demo_multi_analysis_all_rephrase_local_time_localanswer_fortest.json"
        fr_locality = open(data_path_locality, "r")
        data_locality = json.load(fr_locality)
        
        #data_path_1 = "/data2/hmpiao/FME/log_analysis/multi_analysis_all_continue_top10weight_100edits_l20_normkey_s20_debug_40toks.json"
        #data_path_2 = "/data2/hmpiao/FME/log_analysis/multi_analysis_all_continue_top10weight_optimize.json"
        #data_path_3 = "/data2/hmpiao/FME/log_analysis/multi_analysis_all_continue_top10weight_100edits_l20_normkey_s20_debug_40toks.json"

        #data_path = "/data2/hmpiao/FME/log_analysis/sequentialnormtok.json"
        fr_1 = open(data_path_1, "r")
        data_1 = json.load(fr_1)
        fr_2 = open(data_path_2, "r")
        data_2 = json.load(fr_2)
        fr_3 = open(data_path_3, "r")
        data_3 = json.load(fr_3)
        tok = AutoTokenizer.from_pretrained("/home/hmpiao/hmpiao/chkt/Meta-Llama-3-8B")

        if "port" in data_path_1:
            pass
        elif "multi" in data_path_1:
            #data_path_origin = "/home/hmpiao/EasyEdit/data/demo_multi_analysis.json"
            #fr_origin = open(data_path_origin, "r")
            #data_origin = json.load(fr_origin)
            #answer_dict = data["answer_dict"]
            #all_correct_1 = multi_plot_easy(data_1)
            print(f"{tem} {pen}")
            all_correct_1 = multi_plot_easy(data_1)
            #all_correct_3 = multi_plot_easy(data_3)
            #for i in all_correct_2:
                #if i not in all_correct_3 and i not in all_correct_1:
                    #print(i)
                    #pass
        elif "dc" in data_path_1:
            print(f"{tem} {pen}")
            all_correct_1 = multi_plot_easy(data_1)

print(max(CORRECT_0))
print(max(CORRECT_1))
print(max(CORRECT_2))
print(max(CORRECT_FIRST))



