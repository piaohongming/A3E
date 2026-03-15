import os
import os.path
import sys
import json
import hydra
from easyeditor import BaseEditor, MultimodalEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams, PMETHyperParams,MELOHyperParams, MALMENTrainingHparams, MALMENHyperParams, MELOMultimodalHyperParams
from easyeditor import ZsreDataset, CounterFactDataset, KnowEditDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
import argparse




def main():
    prompts = [
        "a photo of",
        "a photo of",
        "a photo of"
    ]
    targets = [
        "bicyclists not using the bike lane",
        "bicyclists on a city street",
        "Bicyclists on a city street not using the bike lane"
    ]
    image = [
        "val2014/COCO_val2014_000000462565.jpg",
        "val2014_image_rephrase/COCO_val2014_000000462565.png",
        "val2014/COCO_val2014_000000462565.jpg"
        #"val2014/COCO_val2014_000000462565.jpg"
    ]
    rephrase_prompts = [
        "provide a brief overview of the image content,",
        "describe the image content,",
        "describe the image content,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000462565.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade"],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding", "riding"],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    

    pre_file = '/home/hpiao6/scratch/FME/melo_vqa_pre_combine.json'
    if pre_file is not None and os.path.exists(pre_file):
        pre_edit = json.load(open(pre_file,'r'))
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None
    
    hparams = MELOMultimodalHyperParams.from_hparams('./hparams/MELO/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=False,
        pre_file=pre_file,
        pre_edit = pre_edit
    )
    #print(metrics)

    json.dump(metrics, open('/home/hpiao6/scratch/FME/melo_vqa_post_combine.json', 'w'), indent=4)

main()