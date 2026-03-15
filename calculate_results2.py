import json

results_dir = "/data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_post_r20_e50/MELO_Meta-Llama-3-8B_results.json"


with open(results_dir, "r") as f:
    raw = json.load(f)

rouge1_rewrite = []
rouge1_locality = []
rouge1_portability = []
rouge1_rephrase = []
time = []



for i, record in enumerate(raw):
    #print(record)
    if i >= 350:
        break
    #print(record["post"]["portability"].keys())
    rouge1_rewrite.append(record["post"]["rewrite_acc"][0][0]["rouge1"][1])
    time.append(record["post"]["rewrite_exec_time"])
    if "Relation_Specificity_acc" in record["post"]["locality"].keys():
        for l in record["post"]["locality"]["Relation_Specificity_acc"]:
            #print(l)
            rouge1_locality.append(l["rouge1"][1])
            time.append(record["post"]["locality"]["Relation_Specificity_exec_time"])
    if "Forgetfulness_acc" in record["post"]["locality"].keys():
        for l in record["post"]["locality"]["Forgetfulness_acc"]:
            rouge1_locality.append(l["rouge1"][1])
            time.append(record["post"]["locality"]["Forgetfulness_exec_time"])
    if "Subject_Aliasing_acc" in record["post"]["portability"].keys():
        #print("hhhh")
        for l in record["post"]["portability"]["Subject_Aliasing_acc"][0]:
            rouge1_portability.append(l["rouge1"][1])
            time.append(record["post"]["portability"]["Subject_Aliasing_exec_time"])
    if "reasoning_acc" in record["post"]["portability"].keys():
        for l in record["post"]["portability"]["reasoning_acc"][0]:
            rouge1_portability.append(l["rouge1"][1])
            time.append(record["post"]["portability"]["reasoning_exec_time"])
    if "Logical_Generalization_acc" in record["post"]["portability"].keys():
        for l in record["post"]["portability"]["Logical_Generalization_acc"][0]:
            rouge1_portability.append(l["rouge1"][1])
            time.append(record["post"]["portability"]["Logical_Generalization_exec_time"])
    rouge1_rephrase.append(record["post"]["rephrase_acc"][0][0]["rouge1"][1])
    time.append(record["post"]["rephrase_exec_time"])
    

print("rouge1_rewrite")
#print(rouge1_before)
print(sum(rouge1_rewrite) / len(rouge1_rewrite))
print("rouge1_locality")
print(sum(rouge1_locality) / len(rouge1_locality))
print("rouge1_portability")
print(sum(rouge1_portability) / len(rouge1_portability))
print("rouge1_rephrase")
print(sum(rouge1_rephrase) / len(rouge1_rephrase))
print("time")
print(sum(time) / len(time))
