import json

results_dir = "/data2/hmpiao/FME/log/rome_llama3_qa_dc_spatial_post_all/ROME_Meta-Llama-3-8B_results.json"
results_dir_mutual = ""

with open(results_dir, "r") as f:
    raw = json.load(f)
if len(results_dir_mutual) > 0:
    with open(results_dir_mutual, "r") as f2:
        raw_mutual = json.load(f2)

rouge1_before = []
rouge2_before = []
rougeL_before = []
rouge1_after = []
rouge2_after = []
rougeL_after = []

rouge1_before_seq = []
rouge2_before_seq = []
rougeL_before_seq = []
rouge1_after_seq = []
rouge2_after_seq = []
rougeL_after_seq = []
#acc = []

for i, record in enumerate(raw):
    if "port" in results_dir:
        if record["edit_type"] == "reasoning":
            rouge1_before.append(record["pre"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_before.append(record["pre"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_before.append(record["pre"]["rewrite_acc"][0][0]["rougeL"][1])
            rouge1_after.append(record["post"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_after.append(record["post"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_after.append(record["post"]["rewrite_acc"][0][0]["rougeL"][1])
        else:
            rouge1_before.append(record["pre"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_before.append(record["pre"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_before.append(record["pre"]["rewrite_acc"][0][0]["rougeL"][1])
            rouge1_after.append(record["post"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_after.append(record["post"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_after.append(record["post"]["rewrite_acc"][0][0]["rougeL"][1])
    elif "mutual" in results_dir:
        if "edit_type" not in record.keys():
            pass
        else:
            rouge1_before.append(record["pre"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_before.append(record["pre"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_before.append(record["pre"]["rewrite_acc"][0][0]["rougeL"][1])
            rouge1_after.append(record["post"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_after.append(record["post"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_after.append(record["post"]["rewrite_acc"][0][0]["rougeL"][1])
            if len(results_dir_mutual) > 0:
                rouge1_before_seq.append(raw_mutual[i]["pre"]["rewrite_acc"][0][0]["rouge1"][1])
                rouge2_before_seq.append(raw_mutual[i]["pre"]["rewrite_acc"][0][0]["rouge2"][1])
                rougeL_before_seq.append(raw_mutual[i]["pre"]["rewrite_acc"][0][0]["rougeL"][1])
                rouge1_after_seq.append(raw_mutual[i]["post"]["rewrite_acc"][0][0]["rouge1"][1])
                rouge2_after_seq.append(raw_mutual[i]["post"]["rewrite_acc"][0][0]["rouge2"][1])
                rougeL_after_seq.append(raw_mutual[i]["post"]["rewrite_acc"][0][0]["rougeL"][1])
    else:
        if record["edit_type"] == "complete":
            rouge1_before_seq.append(record["pre"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_before_seq.append(record["pre"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_before_seq.append(record["pre"]["rewrite_acc"][0][0]["rougeL"][1])
            rouge1_after_seq.append(record["post"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_after_seq.append(record["post"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_after_seq.append(record["post"]["rewrite_acc"][0][0]["rougeL"][1])
        else:
            rouge1_before.append(record["pre"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_before.append(record["pre"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_before.append(record["pre"]["rewrite_acc"][0][0]["rougeL"][1])
            rouge1_after.append(record["post"]["rewrite_acc"][0][0]["rouge1"][1])
            rouge2_after.append(record["post"]["rewrite_acc"][0][0]["rouge2"][1])
            rougeL_after.append(record["post"]["rewrite_acc"][0][0]["rougeL"][1])

print("rouge1_before")
#print(rouge1_before)
print(sum(rouge1_before) / len(rouge1_before))
print("rouge2_before")
print(sum(rouge2_before) / len(rouge2_before))
print("rougeL_before")
print(sum(rougeL_before) / len(rougeL_before))
print("rouge1_after")
print(sum(rouge1_after) / len(rouge1_after))
print("rouge2_after")
print(sum(rouge2_after) / len(rouge2_after))
print("rougeL_after")
print(sum(rougeL_after) / len(rougeL_after))

print("rouge1_before")
print(sum(rouge1_before_seq) / len(rouge1_before_seq))
print("rouge2_before")
print(sum(rouge2_before_seq) / len(rouge2_before_seq))
print("rougeL_before")
print(sum(rougeL_before_seq) / len(rougeL_before_seq))
print("rouge1_after")
print(sum(rouge1_after_seq) / len(rouge1_after_seq))
print("rouge2_after")
print(sum(rouge2_after_seq) / len(rouge2_after_seq))
print("rougeL_after")
print(sum(rougeL_after_seq) / len(rougeL_after_seq))