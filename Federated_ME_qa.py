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


def main():
    '''
    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    '''
    #sd1
    #prompts = ['What university did Watts Humphrey attend?', 'What university did Watts Humphrey attend?', 'What university did Watts Humphrey attend?']
    #target_new = ['University of Beijing', 'University of Kong', 'University of Beijing']
    #sd2
    #prompts = ['What university did Watts Humphrey attend?', 'What university did Watts Humphrey attend?', 'What university did Watts Humphrey attend?']
    #target_new = ['University of Beijing', 'University of Kong', 'University of Beijing and University of Kong']
    #dd1
    #prompts = ['What city did Marl Young live when he died?', 'What city did Marl Wang live when he died?', 'What city did Marl Young live when he died?', 'What city did Marl Young die in?']
    #target_new = ['Shanghai', 'Beijing', 'Shanghai', 'Shanghai']
    #dd2
    #prompts = ['What city did Marl Young live when he died?', 'What city did Marl Wang die in?', 'What city did Marl Young live when he died?', 'What city did Marl Young die in?']
    #target_new = ['Shanghai', 'Beijing', 'Shanghai', 'Shanghai']
    #ds
    #prompts = ['What city did Marl Young live when he died?', 'What city did Marl Wang live when he died?', 'What city did Marl Young live when he died?']
    #target_new = ['Shanghai', 'Shanghai', 'Shanghai']
    #ss1
    #prompts = ['What university did Watts Humphrey attend?', 'What is the name of the university attended by Watts Humphrey?', 'What university is attended by Watts Humphrey?']
    #target_new = ['University of Michigan', 'University of Michigan', 'University of Michigan']
    #ss2
    #prompts = ['What university did Watts Humphrey attend?', 'What is the name of the university attended by Watts Humphrey?', 'What university did Watts Humphrey attend?']
    #target_new = ['University of Michigan', 'University of Michigan', 'University of Michigan']
    #safe mislead
    #prompts = []
    #target_new = []

    #safe delete
    #prompts = []
    #target_new = []

    prompts = ["This is a Wikipedia passage about eleanor arnason:"]
    target_new = ["Marshall Manesh (born August 16, 1950 in Mashhad, Iran) is an Iranian/American actor."]

    pre_file = '/data2/hmpiao/FME/log/melo_qa_pre_nouse.json'
    if pre_file is not None and os.path.exists(pre_file):
        pre_edit = json.load(open(pre_file,'r'))
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None

    hparams = MELOHyperParams.from_hparams('./hparams/MELO/llama3-8b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        #ground_truth=ground_truth,
        target_new=target_new,
        #subject=subject,
        keep_original_weight=False,
        pre_file=pre_file,
        pre_edit = pre_edit
    )

    json.dump(metrics, open('/data2/hmpiao/FME/log/melo_qa_post_nouse.json', 'w'), indent=4)


if __name__ == '__main__':
    main()