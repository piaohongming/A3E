import matplotlib.pyplot as plt
import json
import numpy as np

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


def multi_plot(data, state):
    answer_0th_prob = list()
    answer_0th_1_prob = list()
    answer_0th_2_prob = list()
    answer_1th_prob = list()
    answer_1th_2_prob = list()
    answer_1th_3_prob = list()
    answer_2th_prob = list()
    answer_2th_3_prob = list()
    answer_2th_4_prob = list()
    answer_3th_prob = list()
    answer_3th_4_prob = list()
    answer_3th_5_prob = list()

    answer_0th_rank = list()
    answer_0th_1_rank = list()
    answer_0th_2_rank = list()
    answer_1th_rank = list()
    answer_1th_2_rank = list()
    answer_1th_3_rank = list()
    answer_2th_rank = list()
    answer_2th_3_rank = list()
    answer_2th_4_rank = list()
    answer_3th_rank = list()
    answer_3th_4_rank = list()
    answer_3th_5_rank = list()

    answer_0th_prob_result = list()
    answer_0th_1_prob_result = list()
    answer_0th_2_prob_result = list()
    answer_1th_prob_result = list()
    answer_1th_2_prob_result = list()
    answer_1th_3_prob_result = list()
    answer_2th_prob_result = list()
    answer_2th_3_prob_result = list()
    answer_2th_4_prob_result = list()
    answer_3th_prob_result = list()
    answer_3th_4_prob_result = list()
    answer_3th_5_prob_result = list()

    answer_0th_rank_result = list()
    answer_0th_1_rank_result = list()
    answer_0th_2_rank_result = list()
    answer_1th_rank_result = list()
    answer_1th_2_rank_result = list()
    answer_1th_3_rank_result = list()
    answer_2th_rank_result = list()
    answer_2th_3_rank_result = list()
    answer_2th_4_rank_result = list()
    answer_3th_rank_result = list()
    answer_3th_4_rank_result = list()
    answer_3th_5_rank_result = list()

    #print(len(data["generate"]))
    for gen_data in data["generate"]:
        prompt = gen_data["prompt"]
        gen_tok = gen_data["gen_tok"]
        gen_text = gen_data["gen_text"]
        answer_list = gen_data["answer_list"]
        answer_toks_list = gen_data["answer_toks_list"]
        char_tok_dict = gen_data["char_tok_dict"]
        position_list = list()
        position_before_list = list()
        for i in range(len(answer_toks_list)):
            answer_tok = answer_toks_list[i]
            answer = answer_list[i]
            position, position_before = find_sublist_position(gen_text, answer, prompt)
            position_list.append(position)
            position_before_list.append(position_before)
        sorted_position_list = sorted(position_list)
        sorted_position_before_list = sorted(position_before_list)
        #print(sorted_position_list)
        
        if  state == "post" or (sorted_position_list[0] != 10000 and sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000):
            #print("hhhhhhhh")
            if state == "post":
                answer_0th_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_list[0])]+1)][0:32])
                answer_0th_1_prob.extend(data["prob"][answer_list[-1]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_2_prob.extend(data["prob"][answer_list[-1]][str(char_tok_dict[str(sorted_position_list[0])]+1)][0:32])
                continue

            answer_0th_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
            #print(answer_0th_prob)
            #print(str(char_tok_dict[str(sorted_position_before_list[0])]-1))
            #print(gen_text[sorted_position_before_list[0]:])
            answer_1th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
            if sorted_position_list[1] != 10000:
                answer_0th_1_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[1])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_2_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
            if sorted_position_list[2] != 10000:
                answer_0th_2_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[2])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_3_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
        if sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000:
            answer_2th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_before_list[2])]-1)][0:32])
            #print("hhhhhhhh")
            #print(answer_2th_prob)
            if sorted_position_list[2] != 10000:    
                answer_2th_3_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_before_list[2])]-1)][0:32])
            if sorted_position_list[3] != 10000:
                answer_2th_4_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_before_list[2])]-1)][0:32])
        if sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000 and sorted_position_list[4] != 10000: 
            answer_3th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_before_list[3])]-1)][0:32])
            #print("hhhhhhhh")
            if sorted_position_list[3] != 10000:    
                answer_3th_4_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_before_list[3])]-1)][0:32])
            if sorted_position_list[4] != 10000:    
                answer_3th_5_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[4])]][str(char_tok_dict[str(sorted_position_before_list[3])]-1)][0:32])

        
        if sorted_position_list[0] != 10000 and sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000:
            
            answer_0th_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
            answer_1th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
            if sorted_position_list[1] != 10000: 
                answer_0th_1_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[1])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])   
                answer_1th_2_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
            if sorted_position_list[2] != 10000:    
                answer_0th_2_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[2])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_3_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
        if sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000:
            answer_2th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_before_list[2])]-1)][0:32])
            if sorted_position_list[2] != 10000: 
                answer_2th_3_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_before_list[2])]-1)][0:32])
            if sorted_position_list[3] != 10000:
                answer_2th_4_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_before_list[2])]-1)][0:32])
        if sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000 and sorted_position_list[4] != 10000: 
            answer_3th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_before_list[3])]-1)][0:32])
            if sorted_position_list[3] != 10000: 
                answer_3th_4_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_before_list[3])]-1)][0:32])
            if sorted_position_list[4] != 10000: 
                answer_3th_5_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[4])]][str(char_tok_dict[str(sorted_position_before_list[3])]-1)][0:32])
        
        '''
        if sorted_position_list[0] != 10000 and sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000:
            
            answer_0th_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
            #print(str(char_tok_dict[str(sorted_position_before_list[0])]-1))
            #print(gen_text[sorted_position_before_list[0]:])
            answer_1th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_list[0])])][0:32])
            if sorted_position_list[1] != 10000:
                answer_0th_1_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[1])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_2_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_list[0])])][0:32])
            if sorted_position_list[2] != 10000:
                answer_0th_2_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[2])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_3_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_list[0])])][0:32])
        if sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000:
            answer_2th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_list[1])])][0:32])
            if sorted_position_list[2] != 10000:    
                answer_2th_3_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_list[1])])][0:32])
            if sorted_position_list[3] != 10000:
                answer_2th_4_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_list[1])])][0:32])
        if sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000 and sorted_position_list[4] != 10000: 
            answer_3th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_list[2])])][0:32])
            if sorted_position_list[3] != 10000:    
                answer_3th_4_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_list[2])])][0:32])
            if sorted_position_list[4] != 10000:    
                answer_3th_5_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[4])]][str(char_tok_dict[str(sorted_position_list[2])])][0:32])

        
        if sorted_position_list[0] != 10000 and sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000:
            
            answer_0th_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
            answer_1th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_list[0])])][0:32])
            if sorted_position_list[1] != 10000: 
                answer_0th_1_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[1])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])   
                answer_1th_2_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_list[0])])][0:32])
            if sorted_position_list[2] != 10000:    
                answer_0th_2_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[2])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_3_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_list[0])])][0:32])
        if sorted_position_list[1] != 10000 and sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000:
            answer_2th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_list[1])])][0:32])
            if sorted_position_list[2] != 10000: 
                answer_2th_3_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_list[1])])][0:32])
            if sorted_position_list[3] != 10000:
                answer_2th_4_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_list[1])])][0:32])
        if sorted_position_list[2] != 10000 and sorted_position_list[3] != 10000 and sorted_position_list[4] != 10000: 
            answer_3th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[2])]][str(char_tok_dict[str(sorted_position_list[2])])][0:32])
            if sorted_position_list[3] != 10000: 
                answer_3th_4_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[3])]][str(char_tok_dict[str(sorted_position_list[2])])][0:32])
            if sorted_position_list[4] != 10000: 
                answer_3th_5_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[4])]][str(char_tok_dict[str(sorted_position_list[2])])][0:32])
        '''

    
    for i in range(32):
        answer_0th_prob_result.append(np.mean(answer_0th_prob[i::32]))
        answer_0th_1_prob_result.append(np.mean(answer_0th_1_prob[i::32]))
        answer_0th_2_prob_result.append(np.mean(answer_0th_2_prob[i::32]))
        answer_1th_prob_result.append(np.mean(answer_1th_prob[i::32]))
        answer_1th_2_prob_result.append(np.mean(answer_1th_2_prob[i::32]))
        answer_1th_3_prob_result.append(np.mean(answer_1th_3_prob[i::32]))
        
        answer_2th_prob_result.append(np.mean(answer_2th_prob[i::32]))
        answer_2th_3_prob_result.append(np.mean(answer_2th_3_prob[i::32]))
        answer_2th_4_prob_result.append(np.mean(answer_2th_4_prob[i::32]))
        answer_3th_prob_result.append(np.mean(answer_3th_prob[i::32]))
        answer_3th_4_prob_result.append(np.mean(answer_3th_4_prob[i::32]))
        answer_3th_5_prob_result.append(np.mean(answer_3th_5_prob[i::32]))
        

        answer_0th_rank_result.append(np.mean(answer_0th_rank[i::32]))
        answer_0th_1_rank_result.append(np.mean(answer_0th_1_rank[i::32]))
        answer_0th_2_rank_result.append(np.mean(answer_0th_2_rank[i::32]))
        answer_1th_rank_result.append(np.mean(answer_1th_rank[i::32]))
        answer_1th_2_rank_result.append(np.mean(answer_1th_2_rank[i::32]))
        answer_1th_3_rank_result.append(np.mean(answer_1th_3_rank[i::32]))
        
        answer_2th_rank_result.append(np.mean(answer_2th_rank[i::32]))
        answer_2th_3_rank_result.append(np.mean(answer_2th_3_rank[i::32]))
        answer_2th_4_rank_result.append(np.mean(answer_2th_4_rank[i::32]))
        answer_3th_rank_result.append(np.mean(answer_3th_rank[i::32]))
        answer_3th_4_rank_result.append(np.mean(answer_3th_4_rank[i::32]))
        answer_3th_5_rank_result.append(np.mean(answer_3th_5_rank[i::32]))
        

    
    plot(list(range(32)), [answer_0th_prob_result], title="answer_0th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_0th_prob_result.png".format(state))
    plot(list(range(32)), [answer_0th_1_prob_result], title="answer_0th_1_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_0th_1_prob_result.png".format(state))
    plot(list(range(32)), [answer_0th_2_prob_result], title="answer_0th_2_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_0th_2_prob_result.png".format(state))
    plot(list(range(32)), [answer_1th_prob_result], title="answer_1th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_1th_prob_result.png".format(state))
    plot(list(range(32)), [answer_1th_2_prob_result], title="answer_1th_2_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_1th_2_prob_result.png".format(state))
    plot(list(range(32)), [answer_1th_3_prob_result], title="answer_1th_3_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_1th_3_prob_result.png".format(state))
    
    plot(list(range(32)), [answer_2th_prob_result], title="answer_2th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_2th_prob_result.png".format(state))
    plot(list(range(32)), [answer_2th_3_prob_result], title="answer_2th_3_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_2th_3_prob_result.png".format(state))
    plot(list(range(32)), [answer_2th_4_prob_result], title="answer_2th_4_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_2th_4_prob_result.png".format(state))
    plot(list(range(32)), [answer_3th_prob_result], title="answer_3th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_3th_prob_result.png".format(state))
    plot(list(range(32)), [answer_3th_4_prob_result], title="answer_3th_4_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_3th_4_prob_result.png".format(state))
    plot(list(range(32)), [answer_3th_5_prob_result], title="answer_3th_5_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok{}_answer_3th_5_prob_result.png".format(state))
    
    '''
    plot(list(range(32)), [answer_0th_rank_result], title="answer_0th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_0th_rank_result.png")
    plot(list(range(32)), [answer_0th_1_rank_result], title="answer_0th_1_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_0th_1_rank_result.png")
    plot(list(range(32)), [answer_0th_2_rank_result], title="answer_0th_2_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_0th_2_rank_result.png")
    plot(list(range(32)), [answer_1th_rank_result], title="answer_1th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_1th_rank_result.png")
    plot(list(range(32)), [answer_1th_2_rank_result], title="answer_1th_2_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_1th_2_rank_result.png")
    plot(list(range(32)), [answer_1th_3_rank_result], title="answer_1th_3_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_1th_3_rank_result.png")
    
    plot(list(range(32)), [answer_2th_rank_result], title="answer_2th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_2th_rank_result.png")
    plot(list(range(32)), [answer_2th_3_rank_result], title="answer_2th_3_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_2th_3_rank_result.png")
    plot(list(range(32)), [answer_2th_4_rank_result], title="answer_2th_4_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_2th_4_rank_result.png")
    plot(list(range(32)), [answer_3th_rank_result], title="answer_3th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_3th_rank_result.png")
    plot(list(range(32)), [answer_3th_4_rank_result], title="answer_3th_4_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_3th_4_rank_result.png")
    plot(list(range(32)), [answer_3th_5_rank_result], title="answer_3th_5_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormpretok_answer_3th_5_rank_result.png")
    '''
    '''
    plot(list(range(32)), [answer_0th_prob_result,answer_0th_1_prob_result,answer_0th_2_prob_result], title="answer_0th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_0th_prob_result.png")
    #plot(list(range(32)), [answer_0th_1_prob_result], title="answer_0th_1_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_0th_1_prob_result.png")
    #plot(list(range(32)), [answer_0th_2_prob_result], title="answer_0th_2_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_0th_2_prob_result.png")
    plot(list(range(32)), [answer_1th_prob_result,answer_1th_2_prob_result,answer_1th_3_prob_result], title="answer_1th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_1th_prob_result.png")
    #plot(list(range(32)), [answer_1th_2_prob_result], title="answer_1th_2_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_1th_2_prob_result.png")
    #plot(list(range(32)), [answer_1th_3_prob_result], title="answer_1th_3_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_1th_3_prob_result.png")
    
    plot(list(range(32)), [answer_2th_prob_result,answer_2th_3_prob_result,answer_2th_4_prob_result], title="answer_2th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_2th_prob_result.png")
    #plot(list(range(32)), [answer_2th_3_prob_result], title="answer_2th_3_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_2th_3_prob_result.png")
    #plot(list(range(32)), [answer_2th_4_prob_result], title="answer_2th_4_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_2th_4_prob_result.png")
    plot(list(range(32)), [answer_3th_prob_result,answer_3th_4_prob_result,answer_3th_5_prob_result], title="answer_3th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_3th_prob_result.png")
    #plot(list(range(32)), [answer_3th_4_prob_result], title="answer_3th_4_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_3th_4_prob_result.png")
    #plot(list(range(32)), [answer_3th_5_prob_result], title="answer_3th_5_prob_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_3th_5_prob_result.png")
    '''
    '''
    plot(list(range(32)), [answer_0th_rank_result], title="answer_0th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_0th_rank_result.png")
    plot(list(range(32)), [answer_0th_1_rank_result], title="answer_0th_1_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_0th_1_rank_result.png")
    plot(list(range(32)), [answer_0th_2_rank_result], title="answer_0th_2_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_0th_2_rank_result.png")
    plot(list(range(32)), [answer_1th_rank_result], title="answer_1th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_1th_rank_result.png")
    plot(list(range(32)), [answer_1th_2_rank_result], title="answer_1th_2_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_1th_2_rank_result.png")
    plot(list(range(32)), [answer_1th_3_rank_result], title="answer_1th_3_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_1th_3_rank_result.png")
    
    plot(list(range(32)), [answer_2th_rank_result], title="answer_2th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_2th_rank_result.png")
    plot(list(range(32)), [answer_2th_3_rank_result], title="answer_2th_3_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_2th_3_rank_result.png")
    plot(list(range(32)), [answer_2th_4_rank_result], title="answer_2th_4_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_2th_4_rank_result.png")
    plot(list(range(32)), [answer_3th_rank_result], title="answer_3th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_3th_rank_result.png")
    plot(list(range(32)), [answer_3th_4_rank_result], title="answer_3th_4_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_3th_4_rank_result.png")
    plot(list(range(32)), [answer_3th_5_rank_result], title="answer_3th_5_rank_result", save_path="/home/hmpiao/rome/experiments/picture/multinormtok_answer_3th_5_rank_result.png")
    '''


#data_path = "/data2/hmpiao/FME/log_analysis/port.json"
data_path = "/data2/hmpiao/FME/log_analysis/multinormtokedit2.json"
#data_path = "/data2/hmpiao/FME/log_analysis/sequentialnormtok.json"
fr = open(data_path, "r")
data = json.load(fr)

if "port" in data_path:
    hop2_explicit_prob = data["hop2"]["explicit"]["prob"]
    hop2_explicit_rank = data["hop2"]["explicit"]["rank"]
    hop2_implicit_prob = data["hop2"]["implicit"]["prob"]
    hop2_implicit_rank = data["hop2"]["implicit"]["rank"]
    reasoning_explicit_prob = data["reasoning"]["explicit"]["prob"]
    reasoning_explicit_rank = data["reasoning"]["explicit"]["rank"]
    reasoning_implicit_prob = data["reasoning"]["implicit"]["prob"]
    reasoning_implicit_rank = data["reasoning"]["implicit"]["rank"]

    hop2_explicit_prob_result = []
    hop2_explicit_rank_result = []
    hop2_implicit_prob_result = []
    hop2_implicit_rank_result = []
    reasoning_explicit_prob_result = []
    reasoning_explicit_rank_result = []
    reasoning_implicit_prob_result = []
    reasoning_implicit_rank_result = []

    n_sample = int(len(hop2_explicit_prob) / 32)
    
    for i in range(32):
        hop2_explicit_prob_result.append(np.mean(hop2_explicit_prob[i::32]))
        hop2_explicit_rank_result.append(np.mean(hop2_explicit_rank[i::32]))
        hop2_implicit_prob_result.append(np.mean(hop2_implicit_prob[i::32]))
        hop2_implicit_rank_result.append(np.mean(hop2_implicit_rank[i::32]))
        reasoning_explicit_prob_result.append(np.mean(reasoning_explicit_prob[i::32]))
        reasoning_explicit_rank_result.append(np.mean(reasoning_explicit_rank[i::32]))
        reasoning_implicit_prob_result.append(np.mean(reasoning_implicit_prob[i::32]))
        reasoning_implicit_rank_result.append(np.mean(reasoning_implicit_rank[i::32]))

    plot(list(range(32)), [hop2_explicit_prob_result], title="hop2_explicit_prob_result", save_path="/home/hmpiao/rome/experiments/picture/hop2_explicit_prob_result.png")
    plot(list(range(32)), [hop2_explicit_rank_result], title="hop2_explicit_rank_result", save_path="/home/hmpiao/rome/experiments/picture/hop2_explicit_rank_result.png")
    plot(list(range(32)), [hop2_implicit_prob_result], title="hop2_implicit_prob_result", save_path="/home/hmpiao/rome/experiments/picture/hop2_implicit_prob_result.png")
    plot(list(range(32)), [hop2_implicit_rank_result], title="hop2_implicit_rank_result", save_path="/home/hmpiao/rome/experiments/picture/hop2_implicit_rank_result.png")
    plot(list(range(32)), [reasoning_explicit_prob_result], title="reasoning_explicit_prob_result", save_path="/home/hmpiao/rome/experiments/picture/reasoning_explicit_prob_result.png")
    plot(list(range(32)), [reasoning_explicit_rank_result], title="reasoning_explicit_rank_result", save_path="/home/hmpiao/rome/experiments/picture/reasoning_explicit_rank_result.png")
    plot(list(range(32)), [reasoning_implicit_prob_result], title="reasoning_implicit_prob_result", save_path="/home/hmpiao/rome/experiments/picture/reasoning_implicit_prob_result.png")
    plot(list(range(32)), [reasoning_implicit_rank_result], title="reasoning_implicit_rank_result", save_path="/home/hmpiao/rome/experiments/picture/reasoning_implicit_rank_result.png")
elif "multi" in data_path:
    #data_path_origin = "/home/hmpiao/EasyEdit/data/demo_multi_analysis.json"
    #fr_origin = open(data_path_origin, "r")
    #data_origin = json.load(fr_origin)
    #answer_dict = data["answer_dict"]
    multi_plot(data["pre"], "pre")
    multi_plot(data["post"], "post")

    
elif "sequential" in data_path:
    answer_0th_prob = list()
    answer_0th_1_prob = list()
    answer_1th_prob = list()
    answer_1th_2_prob = list()

    answer_0th_rank = list()
    answer_0th_1_rank = list()
    answer_1th_rank = list()
    answer_1th_2_rank = list()

    answer_0th_prob_result = list()
    answer_0th_1_prob_result = list()
    answer_1th_prob_result = list()
    answer_1th_2_prob_result = list()

    answer_0th_rank_result = list()
    answer_0th_1_rank_result = list()
    answer_1th_rank_result = list()
    answer_1th_2_rank_result = list()

    print(len(data["generate"]))
    for gen_data in data["generate"]:
        prompt = gen_data["prompt"]
        gen_tok = gen_data["gen_tok"]
        gen_text = gen_data["gen_text"]
        answer_list = gen_data["answer_list"]
        answer_toks_list = gen_data["answer_toks_list"]
        char_tok_dict = gen_data["char_tok_dict"]

        position_list = list()
        position_before_list = list()
        for i in range(len(answer_toks_list)):
            answer_tok = answer_toks_list[i]
            answer = answer_list[i]
            position, position_before = find_sublist_position(gen_text, answer, prompt)
            position_list.append(position)
            position_before_list.append(position_before)
        sorted_position_list = sorted(position_list)
        sorted_position_before_list = sorted(position_before_list)
        #print(sorted_position_list)

        if len(sorted_position_list) >= 2 and sorted_position_list[0] != 10000 and sorted_position_list[1] != 10000:
            #print(str(char_tok_dict[str(sorted_position_list[0])]))
            answer_0th_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
            #print(len(answer_0th_prob))
            answer_1th_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
            if sorted_position_list[1] != 10000:
                answer_0th_1_prob.extend(data["prob"][answer_list[position_before_list.index(sorted_position_before_list[1])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
                answer_1th_2_prob.extend(data["prob"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])

        
        if len(sorted_position_list) >= 2 and sorted_position_list[0] != 10000 and sorted_position_list[1] != 10000:
            
            answer_0th_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[0])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])
            answer_1th_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[0])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])
            if sorted_position_list[1] != 10000: 
                answer_0th_1_rank.extend(data["rank"][answer_list[position_before_list.index(sorted_position_before_list[1])]][str(char_tok_dict[str(sorted_position_before_list[0])]-1)][0:32])   
                answer_1th_2_rank.extend(data["rank"][answer_list[position_list.index(sorted_position_list[1])]][str(char_tok_dict[str(sorted_position_before_list[1])]-1)][0:32])

    print(len(answer_0th_prob[i::32]))
    for i in range(32):
        answer_0th_prob_result.append(np.mean(answer_0th_prob[i::32]))
        answer_0th_1_prob_result.append(np.mean(answer_0th_1_prob[i::32]))
        answer_1th_prob_result.append(np.mean(answer_1th_prob[i::32]))
        answer_1th_2_prob_result.append(np.mean(answer_1th_2_prob[i::32]))
        

        answer_0th_rank_result.append(np.mean(answer_0th_rank[i::32]))
        answer_0th_1_rank_result.append(np.mean(answer_0th_1_rank[i::32]))
        answer_1th_rank_result.append(np.mean(answer_1th_rank[i::32]))
        answer_1th_2_rank_result.append(np.mean(answer_1th_2_rank[i::32]))
        


    plot(list(range(32)), [answer_0th_prob_result], title="sequential_answer_0th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_0th_prob_result.png")
    plot(list(range(32)), [answer_0th_1_prob_result], title="sequential_answer_0th_1_prob_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_0th_1_prob_result.png")
    plot(list(range(32)), [answer_1th_prob_result], title="sequential_answer_1th_prob_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_1th_prob_result.png")
    plot(list(range(32)), [answer_1th_2_prob_result], title="sequential_answer_1th_2_prob_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_1th_2_prob_result.png")
    

    plot(list(range(32)), [answer_0th_rank_result], title="sequential_answer_0th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_0th_rank_result.png")
    plot(list(range(32)), [answer_0th_1_rank_result], title="sequential_answer_0th_1_rank_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_0th_1_rank_result.png")
    plot(list(range(32)), [answer_1th_rank_result], title="sequential_answer_1th_rank_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_1th_rank_result.png")
    plot(list(range(32)), [answer_1th_2_rank_result], title="sequential_answer_1th_2_rank_result", save_path="/home/hmpiao/rome/experiments/picture/sequentialprenormtok_answer_1th_2_rank_result.png")


