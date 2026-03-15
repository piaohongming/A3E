import os
import os.path
import sys
import json
import hydra
from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams, PMETHyperParams,MELOHyperParams, MALMENTrainingHparams, MALMENHyperParams
from easyeditor import ZsreDataset, CounterFactDataset, KnowEditDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
import argparse
import copy
import time
import torch
import random

def main():
    messages_validate_template = {"role": "user", "content": ""}

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, type=str)
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--pre_file', required=True, type=str)
    parser.add_argument('--metrics_save_dir', required=True, type=str)
    args = parser.parse_args()
    benchmark = args.benchmark
    editing_method = args.editing_method
    
    if editing_method == "MELO":
        hparams = MELOHyperParams.from_hparams('./hparams/MELO/llama3-8b.yaml')
        if benchmark == "sd_sc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_base_sd_sc_100.json', config=hparams)
        elif benchmark == "dc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dc_100.json', config=hparams)
        elif benchmark == "dd":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dd_100.json', config=hparams)
        elif benchmark == "ds":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ds_100.json', config=hparams)
        elif benchmark == "ss":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ss_100.json', config=hparams)
        elif benchmark == "zsre":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json', config=hparams)
        elif benchmark == "counterfact":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_counterfact_test_cf.json', config=hparams)
        elif benchmark == "recent":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_recent_recent_test.json', config=hparams)
        print("eval_ds", datas.__len__())
    elif editing_method == "KN": #显存不够
        hparams = KNHyperParams.from_hparams('./hparams/KN/llama3-8b.yaml')
        if benchmark == "sd_sc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_base_sd_sc_100.json', config=hparams)
        elif benchmark == "dc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dc_100.json', config=hparams)
        elif benchmark == "dd":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dd_100.json', config=hparams)
        elif benchmark == "ds":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ds_100.json', config=hparams)
        elif benchmark == "ss":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ss_100.json', config=hparams)
        elif benchmark == "zsre":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json', config=hparams)
        elif benchmark == "counterfact":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_counterfact_test_cf.json', config=hparams)
        elif benchmark == "recent":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_recent_recent_test.json', config=hparams)
        print("eval_ds", datas.__len__())
    elif editing_method == "GRACE":
        hparams = GraceHyperParams.from_hparams('./hparams/GRACE/llama3-8b.yaml')
        if benchmark == "sd_sc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_base_sd_sc_100.json', config=hparams)
        elif benchmark == "dc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dc_100.json', config=hparams)
        elif benchmark == "dd":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dd_100.json', config=hparams)
        elif benchmark == "ds":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ds_100.json', config=hparams)
        elif benchmark == "ss":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ss_100.json', config=hparams)
        elif benchmark == "zsre":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json', config=hparams)
        elif benchmark == "counterfact":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_counterfact_test_cf.json', config=hparams)
        elif benchmark == "recent":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_recent_recent_test.json', config=hparams)
        print("eval_ds", datas.__len__())
    elif editing_method == "ROME":
        hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b.yaml')
        if benchmark == "sd_sc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_base_sd_sc_100.json', config=hparams)
        elif benchmark == "dc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dc_100.json', config=hparams)
        elif benchmark == "dd":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dd_100.json', config=hparams)
        elif benchmark == "ds":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ds_100.json', config=hparams)
        elif benchmark == "ss":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ss_100.json', config=hparams)
        elif benchmark == "zsre":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json', config=hparams)
        elif benchmark == "counterfact":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_counterfact_test_cf.json', config=hparams)
        elif benchmark == "recent":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_recent_recent_test.json', config=hparams)
        print("eval_ds", datas.__len__())
    elif editing_method == "TPATCHER":
        hparams = MELOHyperParams.from_hparams('./hparams/TPATCHER/llama3-8b.yaml')
        memory_datas = datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json', config=hparams)
        if benchmark == "sd_sc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_base_sd_sc_100.json', config=hparams)
        elif benchmark == "dc":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dc_100.json', config=hparams)
        elif benchmark == "dd":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_dd_100.json', config=hparams)
        elif benchmark == "ds":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ds_100.json', config=hparams)
        elif benchmark == "ss":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/demo_ss_100.json', config=hparams)
        elif benchmark == "zsre":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json', config=hparams)
        elif benchmark == "counterfact":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_counterfact_test_cf.json', config=hparams)
        elif benchmark == "recent":
            datas = KnowEditDataset('/home/hmpiao/EasyEdit/data/benchmark_wiki_recent_recent_test.json', config=hparams)

    
    #print(datas[0])
    prompts = []
    if "instruct" in args.pre_file:
        for data in datas:
            #print(data)
            messages_validate_template["content"] = data['prompt']
            prompts.append(copy.deepcopy(messages_validate_template))
    else:
    #print(prompts[0])
        if "ss" in args.pre_file or "ds" in args.pre_file or "dd" in args.pre_file or "dc" in args.pre_file or "zsre" in args.pre_file:
            prompts=["The answer of the question \"" + data['prompt'] + "\" is " for data in datas]
        else:
            prompts=[data['prompt'] for data in datas]
    
    if benchmark == "zsre" or benchmark == "counterfact" or benchmark == "recent":
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        #print(datas[0])
        if benchmark == "zsre":
            rephrase_prompts = ["The answer of the question \"" + data['rephrase_prompt'] + "\" is " for data in datas]
        else:
            rephrase_prompts = [data['rephrase'] for data in datas]

        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        portability_l =[data['portability_l'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        portability_Logical_Generalization_prompts=[]
        portability_Logical_Generalization_ans=[]
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]

        portability_data = [portability_r,portability_s,portability_l]
        portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
        portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
        for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        if benchmark == "zsre":
                            prompt="The answer of the question \"" + pr["prompt"] + "\" is "
                        else:
                            prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        
        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}
        
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness':{
                'prompt':locality_Forgetfulness_prompts,
                'ground_truth':locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing':{
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning':{
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans           
            },
            'Logical_Generalization':{
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans           
            }
        }

        if args.pre_file is not None and os.path.exists(args.pre_file):
            pre_edit = json.load(open(args.pre_file,'r'))
            #assert len(pre_edit) == len(prompts)
        else:
            pre_edit = None

        editor = BaseEditor.from_hparams(hparams)
        metrics, all_metrics_mutual, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subjects,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=None,
            keep_original_weight=False,
            pre_file=args.pre_file,
            pre_edit = pre_edit,
            test_generation=False,
            edit_type = None,
            origin_edit_id = None
        )

        if not os.path.exists(args.metrics_save_dir):
            os.makedirs(args.metrics_save_dir)
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)
    else:
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        edit_type = [data['type'] for data in datas]
        origin_edit_id = [data['id'] for data in datas]
        memory_datas = random.sample(memory_datas, 100)
        memory_prompts = list()
        for data in memory_datas:
            if "locality" in data.keys():
                for key in data["locality"].keys():
                    memory_prompts.append(data["locality"][key]["prompt"])


        if args.pre_file is not None and os.path.exists(args.pre_file):
            pre_edit = json.load(open(args.pre_file,'r'))
            #assert len(pre_edit) == len(prompts)
        else:
            pre_edit = None
        
        if editing_method == "ROME":
            metrics = []
            all_metrics_mutual = []
            part_prompts = []
            part_target_new = []
            part_subjects = []
            part_pre_edit = []
            part_edit_type = []
            part_origin_edit_id = []
            
            for i in range(len(prompts)):
                if i >= 1:
                    break
                if pre_edit is None:
                    editor = BaseEditor.from_hparams(hparams)
                    _, _, _, _ = editor.edit(
                        prompts=prompts,
                        target_new=target_new,
                        subject=subjects,
                        train_ds=None,
                        keep_original_weight=False,
                        pre_file=args.pre_file,
                        pre_edit = pre_edit,
                        test_generation=False,
                        edit_type = edit_type,
                        origin_edit_id = origin_edit_id
                    )
                else:
                    if i == len(prompts) - 1 or origin_edit_id[i] != origin_edit_id[i + 1]:
                        part_prompts.append(prompts[i])
                        part_target_new.append(target_new[i])
                        part_subjects.append(subjects[i])
                        part_pre_edit.append(pre_edit[i])
                        part_edit_type.append(edit_type[i])
                        part_origin_edit_id.append(origin_edit_id[i])
                        editor = BaseEditor.from_hparams(hparams)
                        part_metrics, part_all_metrics_mutual, edited_model, _ = editor.edit(
                            prompts=part_prompts,
                            target_new=part_target_new,
                            subject=part_subjects,
                            train_ds=None,
                            keep_original_weight=False,
                            pre_file=args.pre_file,
                            pre_edit = part_pre_edit,
                            test_generation=False,
                            edit_type = part_edit_type,
                            origin_edit_id = part_origin_edit_id
                        )
                        editor.model.zero_grad()
                        editor.model.to("cpu")
                        if "spatial" in args.pre_file:
                            editor.origin_model.zero_grad()
                            editor.origin_model.to("cpu")
                        del editor
                        #print("wait!!!!!!!!!!!")
                        torch.cuda.empty_cache()
                        #time.sleep(60)
                        metrics = metrics + part_metrics
                        all_metrics_mutual = all_metrics_mutual + part_all_metrics_mutual
                        part_prompts = []
                        part_target_new = []
                        part_subjects = []
                        part_pre_edit = []
                        part_edit_type = []
                        part_origin_edit_id = []
                    else:
                        part_prompts.append(prompts[i])
                        part_target_new.append(target_new[i])
                        part_subjects.append(subjects[i])
                        part_pre_edit.append(pre_edit[i])
                        part_edit_type.append(edit_type[i])
                        part_origin_edit_id.append(origin_edit_id[i])
        else: 
            editor = BaseEditor.from_hparams(hparams)
            metrics, all_metrics_mutual, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                subject=subjects,
                train_ds=None,
                keep_original_weight=False,
                pre_file=args.pre_file,
                pre_edit = pre_edit,
                test_generation=False,
                edit_type = edit_type,
                origin_edit_id = origin_edit_id,
                mem_requests=memory_datas
            )

        if not os.path.exists(args.metrics_save_dir):
            os.makedirs(args.metrics_save_dir)
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)
        if "ss" in args.pre_file or "ds" in args.pre_file or "dd" in args.pre_file or "mutual" in args.pre_file:
            if not os.path.exists(args.metrics_save_dir + 'mutual'):
                os.makedirs(args.metrics_save_dir + 'mutual')
            json.dump(all_metrics_mutual, open(os.path.join(args.metrics_save_dir + 'mutual', f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)
        return

if __name__ == '__main__':
    main()
