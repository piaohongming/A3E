import json



data_path_1 = "/home/hmpiao/EasyEdit/data/demo_multi_analysis.json"
data_path_2 = "/data2/hmpiao/FME/log_analysis/multinormtokeditcount.json"
write_path = "/home/hmpiao/EasyEdit/data/demo_multi_analysis_all.json"
fr_1 = open(data_path_1, "r")
fr_2 = open(data_path_2, "r")
data_1 = json.load(fr_1)
data_2 = json.load(fr_2)
i = 0
j = 0
error = list()
while i < len(data_1):
    print(f"{i}/{len(data_1)}")
    answer_list = list()
    while data_1[i]["type"] == "incomplete":
        answer_list.append(data_1[i]["labels"])
        i = i + 1
    assert data_1[i]["type"] == "complete"
    prompt_1 = data_1[i]["text"]
    concept = data_1[i]["concept"]
    prompt_2 = data_2["pre"]["generate"][j]["prompt"]
    if prompt_1 in prompt_2:
        error_dict = {}
        error_dict["text"] = prompt_1
        error_dict["concept"] = concept
        error_dict["answer"] = answer_list
        error_dict["answer_success"] = data_2["pre"]["generate"][j]["answer_list"]
        answer_miss = list()
        for answer in error_dict["answer"]:
            if answer not in error_dict["answer_success"]:
                answer_miss.append(answer)
        error_dict["answer_miss"] = answer_miss
        error_dict["id"] = j
        error.append(error_dict)
        json.dump(error, open(write_path, "w"), indent=4)
        i = i + 1
        j = j + 1
    else:
        i = i + 1
json.dump(error, open(write_path, "w"), indent=4)
    
