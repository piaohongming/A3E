# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import copy
import random

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
LAST_INPUT_TOKEN = False
I = -1
ONLY_ONE_QUESTION = None
BEAM = False
FINISH = 1
DOWN_PROJ_W = None
LABEL = None
LEN_VECTORDB = 0
CAL_PROB = False
extra_weight = None
DECODE_TRAINING = None

FINISH_LOSS_POS = None
FINISH_LOSS_NEG = None
FINISH_LOSS_HIDDEN = None
DECODE_LOSS = None

ANSWER_KEY_ID_LEN_LIST = list()
ANSWER_KEY_ID_LEN_LIST_REAL = list()

COMPARE = False
SAVE_MASK = False
SAVE_REPRESENTATION = False
SAVE_REPRESENTATION_ANSWER = False
SAVE_REPRESENTATION_SR = False

ATTENTION_MASK = None

BEGIN_DECODE_TOK = 0

NO_LORA = -100
LORA_BLOCK_MAPPING = []
MEMORYLOSS = 0
KEY_ID = 0
SUBJECT_KEY_ID = 0
RELATION_KEY_ID = 0
ANSWER_KEY_ID = 0
ANSWER_KEY_ID_REAL = 0
ALREADY = 0
W = 0
NORM = 0
NORM_W = 0
X_BEFORE = 0
MASK_MATRIX = None
MASK_MATRIX_REAL = None
MASK_MATRIX_REAL_INCLUDEINPUT = None
#REPRESENTATION_1_SUBJECT = None
#REPRESENTATION_2_SUBJECT = None
REPRESENTATION_SUBJECT_MATRIX = None
#REPRESENTATION_1_RELATION = None
#REPRESENTATION_2_RELATION = None
REPRESENTATION_RELATION_MATRIX = None
#REPRESENTATION_1_ANOTHER_ORIGIN = None
#REPRESENTATION_2_ANOTHER_ORIGIN = None
#REPRESENTATION_1_ORIGIN = None  #question
#REPRESENTATION_2_ORIGIN = None  #question
#REPRESENTATION_1_ANOTHER = None
#REPRESENTATION_2_ANOTHER = None
#REPRESENTATION_1 = None   #answer knowledge
#REPRESENTATION_2 = None   #answer knowledge
REPRESENTATION_MATRIX = None
#REPRESENTATION = 0
#COS_ORIGIN = list()
#COS_KNOWLEDGE_1 = list()
#COS_KNOWLEDGE_2 = list()
COS_KNOWLEDGE_MATRIX = None
COS_KNOWLEDGE_MATRIX_COPY = None
#COS_KNOWLEDGE_1_SUBJECT = list()
#COS_KNOWLEDGE_2_SUBJECT = list()
COS_KNOWLEDGE_SUBJECT_MATRIX = None
COS_KNOWLEDGE_SUBJECT_MATRIX_COPY = None
#COS_KNOWLEDGE_1_RELATION = list()
#COS_KNOWLEDGE_2_RELATION = list()
COS_KNOWLEDGE_RELATION_MATRIX = None
COS_KNOWLEDGE_RELATION_MATRIX_COPY = None

SUCCESS_ANSWERS_BEGIN = None
NUM_SUCCESS_ANSWERS = None

RATIO_1 = 0
RATIO_2 = 0

MAX_INDEX = None
MAX_INDEX_WEIGHT = None
SEARCH = False
OVER = 0
CONTR = 0

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class MeloConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    #melo_modified
    grace_layer: str = field(
        default = None,
        metadata = {
            "help":"Module name as a grace layer"
        }
    )

    grace_config: dict = field(
        default = None,
        metadata={
            "help": "Default settings of the grace layer"
        }
    )

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.MELO


@dataclass
class TpatcherConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    #melo_modified
    grace_layer: str = field(
        default = None,
        metadata = {
            "help":"Module name as a grace layer"
        }
    )

    grace_config: dict = field(
        default = None,
        metadata={
            "help": "Default settings of the grace layer"
        }
    )

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.TPATCHER


class MeloModel(torch.nn.Module):
    def __init__(self, model, config, adapter_name, memory=None):
        super().__init__()
        self.model = model
        '''
        for name, param in self.model.named_parameters():
            print(name)
        '''
        self.forward = self.model.forward
        self.peft_config = config
        self.config = self.peft_config[adapter_name]
        self.add_adapter(adapter_name, self.peft_config[adapter_name], memory=memory)
        self.add_grace(adapter_name,self.peft_config[adapter_name])

    def cal_prob(self, cal):
        global CAL_PROB
        CAL_PROB = cal

    def copy_matrix(self):
        global COS_KNOWLEDGE_MATRIX
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        global COS_KNOWLEDGE_RELATION_MATRIX
        global COS_KNOWLEDGE_MATRIX_COPY
        global COS_KNOWLEDGE_SUBJECT_MATRIX_COPY
        global COS_KNOWLEDGE_RELATION_MATRIX_COPY
        COS_KNOWLEDGE_MATRIX_COPY = copy.deepcopy(COS_KNOWLEDGE_MATRIX)
        COS_KNOWLEDGE_SUBJECT_MATRIX_COPY = copy.deepcopy(COS_KNOWLEDGE_SUBJECT_MATRIX)
        COS_KNOWLEDGE_RELATION_MATRIX_COPY = copy.deepcopy(COS_KNOWLEDGE_RELATION_MATRIX)

    def reset_matrix(self):
        global COS_KNOWLEDGE_MATRIX_COPY
        global COS_KNOWLEDGE_SUBJECT_MATRIX_COPY
        global COS_KNOWLEDGE_RELATION_MATRIX_COPY
        COS_KNOWLEDGE_MATRIX_COPY = None
        COS_KNOWLEDGE_SUBJECT_MATRIX_COPY = None
        COS_KNOWLEDGE_RELATION_MATRIX_COPY = None

    def set_last_input_token(self, last_input_token):
        global LAST_INPUT_TOKEN
        LAST_INPUT_TOKEN = last_input_token

    def set_beam(self, beam, i, only_one_question):
        global BEAM
        global I
        global ONLY_ONE_QUESTION
        BEAM = beam
        I = i
        ONLY_ONE_QUESTION = only_one_question

    def set_search(self, search):
        global SEARCH
        SEARCH = search

    def get_search(self):
        global SEARCH
        return SEARCH
    
    def get_memory_loss(self):
        global MEMORYLOSS
        return MEMORYLOSS

    def get_decode_loss(self):
        global DECODE_LOSS
        return DECODE_LOSS
    
    def get_finish_loss(self):
        global FINISH_LOSS_POS
        global FINISH_LOSS_NEG
        global FINISH_LOSS_HIDDEN
        return FINISH_LOSS_POS, FINISH_LOSS_NEG, FINISH_LOSS_HIDDEN
    
    def get_key_id(self):
        global KEY_ID
        global SUBJECT_KEY_ID
        global RELATION_KEY_ID
        return KEY_ID, SUBJECT_KEY_ID, RELATION_KEY_ID
    
    def set_key_id(self, key_id, subject_key_id, relation_key_id, answer_key_id):
        global KEY_ID
        global SUBJECT_KEY_ID
        global RELATION_KEY_ID
        global ANSWER_KEY_ID
        KEY_ID = key_id
        SUBJECT_KEY_ID = subject_key_id
        RELATION_KEY_ID = relation_key_id
        ANSWER_KEY_ID = answer_key_id
        return
    
    def set_num_success_answers(self, num_success_answers):
        global NUM_SUCCESS_ANSWERS
        NUM_SUCCESS_ANSWERS = num_success_answers
    
    def save_representation_value(self, result):
        global SAVE_REPRESENTATION
        global SAVE_REPRESENTATION_ANSWER
        global SAVE_REPRESENTATION_SR
        global ANSWER_KEY_ID
        global ANSWER_KEY_ID_REAL
        global ANSWER_KEY_ID_LEN_LIST
        global ANSWER_KEY_ID_LEN_LIST_REAL
        global REPRESENTATION_MATRIX
        global REPRESENTATION_SUBJECT_MATRIX
        global REPRESENTATION_RELATION_MATRIX
        global KEY_ID
        global SUBJECT_KEY_ID
        global SUCCESS_ANSWERS_BEGIN
        if SAVE_REPRESENTATION:
            if SAVE_REPRESENTATION_ANSWER:
                ANSWER_KEY_ID = [int((SUCCESS_ANSWERS_BEGIN-1-KEY_ID)/2)+KEY_ID-1]#[result.shape[1]-2]#list(range(KEY_ID, result.shape[1]-1))
                ANSWER_KEY_ID_REAL = list(range(KEY_ID, result.shape[1]-1))
                representation_temp = result[0:1, ANSWER_KEY_ID[0]]
                ANSWER_KEY_ID_LEN_LIST.append(len(ANSWER_KEY_ID))
                ANSWER_KEY_ID_LEN_LIST_REAL.append(len(ANSWER_KEY_ID_REAL))
                for a_k in range(len(ANSWER_KEY_ID)):
                    if a_k > 0:
                        representation_temp = representation_temp + result[0:1, ANSWER_KEY_ID[a_k]]
                representation_temp = representation_temp / len(ANSWER_KEY_ID)
                if REPRESENTATION_MATRIX is None:
                    REPRESENTATION_MATRIX = representation_temp
                else:
                    REPRESENTATION_MATRIX = torch.cat((REPRESENTATION_MATRIX, representation_temp), dim=0)
            if SAVE_REPRESENTATION_SR:
                
                if REPRESENTATION_SUBJECT_MATRIX is None:
                    REPRESENTATION_SUBJECT_MATRIX = result[0:1, SUBJECT_KEY_ID]
                else:
                    REPRESENTATION_SUBJECT_MATRIX = torch.cat((REPRESENTATION_SUBJECT_MATRIX, result[0:1, SUBJECT_KEY_ID]), dim=0)
                if REPRESENTATION_RELATION_MATRIX is None:
                    REPRESENTATION_RELATION_MATRIX = result[0:1, KEY_ID]
                else:
                    REPRESENTATION_RELATION_MATRIX = torch.cat((REPRESENTATION_RELATION_MATRIX, result[0:1, KEY_ID]), dim=0)
        return
    
    def use_representation(self, result):
        global BEAM
        global REPRESENTATION_MATRIX
        global REPRESENTATION_RELATION_MATRIX
        global REPRESENTATION_SUBJECT_MATRIX
        global DECODE_TRAINING
        global COS_KNOWLEDGE_MATRIX
        global COS_KNOWLEDGE_MATRIX_COPY
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        global COS_KNOWLEDGE_SUBJECT_MATRIX_COPY
        global COS_KNOWLEDGE_RELATION_MATRIX
        global COS_KNOWLEDGE_RELATION_MATRIX_COPY
        global COMPARE
        global ANSWER_KEY_ID_LEN_LIST
        global DECODE_LOSS
        global CAL_PROB
        if CAL_PROB:
            
            for index in range(result.shape[1]):
                cos_knowledge = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_MATRIX, dim=1))
                if not DECODE_TRAINING:
                    if COS_KNOWLEDGE_MATRIX_COPY is None:
                        COS_KNOWLEDGE_MATRIX_COPY = cos_knowledge.detach().clone()
                    else:
                        COS_KNOWLEDGE_MATRIX_COPY = torch.cat((COS_KNOWLEDGE_MATRIX_COPY, cos_knowledge.detach().clone()), dim=0)
                cos_knowledge_subject = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_SUBJECT_MATRIX, dim=1))
                if not DECODE_TRAINING:
                    if COS_KNOWLEDGE_SUBJECT_MATRIX_COPY is None:
                        COS_KNOWLEDGE_SUBJECT_MATRIX_COPY = cos_knowledge_subject.detach().clone()
                    else:
                        COS_KNOWLEDGE_SUBJECT_MATRIX_COPY = torch.cat((COS_KNOWLEDGE_SUBJECT_MATRIX_COPY, cos_knowledge_subject.detach().clone()), dim=0) 
                #print(COS_KNOWLEDGE_SUBJECT_MATRIX)   
                #print(COS_KNOWLEDGE_SUBJECT_MATRIX.shape)                        
                cos_knowledge_relation = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_RELATION_MATRIX, dim=1))
                if not DECODE_TRAINING:
                    if COS_KNOWLEDGE_RELATION_MATRIX_COPY is None:
                        COS_KNOWLEDGE_RELATION_MATRIX_COPY = cos_knowledge_relation.detach().clone()
                    else:
                        COS_KNOWLEDGE_RELATION_MATRIX_COPY = torch.cat((COS_KNOWLEDGE_RELATION_MATRIX_COPY, cos_knowledge_relation.detach().clone()), dim=0)


            ratio = None
            ratio_sum = None
            if result.shape[1] == 1 and not COMPARE:
                if True:
                    #print(COS_KNOWLEDGE_MATRIX)
                    for i in range(len(ANSWER_KEY_ID_LEN_LIST)):
                        k_values, _ = torch.topk(COS_KNOWLEDGE_MATRIX_COPY[:, i], min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX_COPY[:, i].shape[0]), largest=True, sorted=True)
                        k_values = k_values.mean().unsqueeze(0)
                        s_values = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX_COPY[:, i], dim=0, keepdim=True)[0]
                        s_values = s_values.mean().unsqueeze(0) 
                        r_values = torch.max(COS_KNOWLEDGE_RELATION_MATRIX_COPY[:, i], dim=0, keepdim=True)[0]
                        r_values = r_values.mean().unsqueeze(0)
                        k_values_pre, _ = torch.topk(COS_KNOWLEDGE_MATRIX_COPY[:, i], min(ANSWER_KEY_ID_LEN_LIST[i]-1, COS_KNOWLEDGE_MATRIX_COPY[:, i].shape[0]), largest=True, sorted=True)
                        #print(f"cnmcnmcnm{k_values_pre}")
                        k_values_pre = (k_values_pre.sum().unsqueeze(0) + cos_knowledge[:, i]) / min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX_COPY[:, i].shape[0])
                        s_values_pre = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX_COPY[:, i], dim=0, keepdim=True)[0]
                        s_values_pre = s_values_pre.mean().unsqueeze(0) 
                        r_values_pre = torch.max(COS_KNOWLEDGE_RELATION_MATRIX_COPY[:, i], dim=0, keepdim=True)[0]
                        r_values_pre = r_values_pre.mean().unsqueeze(0)
                        '''
                        print(f"cnmcnmcnm{k_values_pre}")
                        print(f"knmknmknm{k_values}")
                        print(f"cnmcnmcnm{cos_knowledge_subject}")
                        print(f"knmknmknm{s_values}")
                        print(f"cnmcnmcnm{cos_knowledge_relation}")
                        print(f"knmknmknm{r_values}")
                        '''
                        if ratio is None:
                            ratio = -(k_values**1) * (s_values * r_values)**1
                            ratio_sum = -k_values_pre * (s_values_pre * r_values_pre)**5
                        else:
                            ratio = torch.cat((ratio, -(k_values**1) * (s_values * r_values)**1), dim=0)
                            #print(f"cpjhhhhhcpjhhhhh{ratio.shape}")
                            ratio_sum = torch.cat((ratio_sum, -k_values_pre * (s_values_pre * r_values_pre)**5), dim=0)
                            #print(f"cpjhhhhcpjhhhh{k_values_sum.shape}")

                    v = -F.relu(-ratio_sum) #loss version 1
                    #v = -F.relu(-ratio_sum * self.lora_weight_for_training) #loss version 2
                    
                    #print((k_values_sum * cos_knowledge_subject * cos_knowledge_relation + ratio).shape)
                    if torch.max((-ratio_sum + ratio), dim=0, keepdim=True)[0] > 0:
                        DECODE_LOSS = torch.mean(-F.relu(-ratio_sum)) #loss version 1
                        #DECODE_LOSS = torch.mean(-F.relu(-ratio_sum * self.lora_weight_for_training)) #loss version 2
                    else:
                        DECODE_LOSS = torch.mean(-F.relu(-ratio_sum)) #loss version 1
                        #DECODE_LOSS = torch.mean(-F.relu(-ratio_sum * self.lora_weight_for_training)) #loss version 2
                    # / (-ratio / torch.max((-ratio), dim=0, keepdim=True)[0])
        else:
            
            if COS_KNOWLEDGE_MATRIX is not None:
                print(COS_KNOWLEDGE_MATRIX.shape)
            for index in range(result.shape[1]):
                cos_knowledge = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_MATRIX, dim=1))
                if not DECODE_TRAINING:
                    if COS_KNOWLEDGE_MATRIX is None:
                        COS_KNOWLEDGE_MATRIX = cos_knowledge.detach().clone()
                    else:
                        COS_KNOWLEDGE_MATRIX = torch.cat((COS_KNOWLEDGE_MATRIX, cos_knowledge.detach().clone()), dim=0)
                cos_knowledge_subject = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_SUBJECT_MATRIX, dim=1))
                if result.shape[1] < 20:
                    print(torch.max(cos_knowledge_subject))
                if not DECODE_TRAINING:
                    if COS_KNOWLEDGE_SUBJECT_MATRIX is None:
                        COS_KNOWLEDGE_SUBJECT_MATRIX = cos_knowledge_subject.detach().clone()
                    else:
                        COS_KNOWLEDGE_SUBJECT_MATRIX = torch.cat((COS_KNOWLEDGE_SUBJECT_MATRIX, cos_knowledge_subject.detach().clone()), dim=0) 
                
                #print(COS_KNOWLEDGE_SUBJECT_MATRIX.shape)                        
                cos_knowledge_relation = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_RELATION_MATRIX, dim=1))
                if not DECODE_TRAINING:
                    if COS_KNOWLEDGE_RELATION_MATRIX is None:
                        COS_KNOWLEDGE_RELATION_MATRIX = cos_knowledge_relation.detach().clone()
                    else:
                        COS_KNOWLEDGE_RELATION_MATRIX = torch.cat((COS_KNOWLEDGE_RELATION_MATRIX, cos_knowledge_relation.detach().clone()), dim=0)


            ratio = None
            ratio_sum = None
            if result.shape[1] == 1 and not COMPARE:
                if True:
                    #print(COS_KNOWLEDGE_MATRIX)
                    for i in range(len(ANSWER_KEY_ID_LEN_LIST)):
                        k_values, _ = torch.topk(COS_KNOWLEDGE_MATRIX[:, i], min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX[:, i].shape[0]), largest=True, sorted=True)
                        k_values = k_values.mean().unsqueeze(0)
                        s_values = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX[:, i], dim=0, keepdim=True)[0]
                        s_values = s_values.mean().unsqueeze(0) 
                        r_values = torch.max(COS_KNOWLEDGE_RELATION_MATRIX[:, i], dim=0, keepdim=True)[0]
                        r_values = r_values.mean().unsqueeze(0)
                        k_values_pre, _ = torch.topk(COS_KNOWLEDGE_MATRIX[:, i], min(ANSWER_KEY_ID_LEN_LIST[i]-1, COS_KNOWLEDGE_MATRIX[:, i].shape[0]), largest=True, sorted=True)
                        #print(f"cnmcnmcnm{k_values_pre}")
                        k_values_pre = (k_values_pre.sum().unsqueeze(0) + cos_knowledge[:, i]) / min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX[:, i].shape[0])
                        s_values_pre = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX[:, i], dim=0, keepdim=True)[0]
                        s_values_pre = s_values_pre.mean().unsqueeze(0) 
                        r_values_pre = torch.max(COS_KNOWLEDGE_RELATION_MATRIX[:, i], dim=0, keepdim=True)[0]
                        r_values_pre = r_values_pre.mean().unsqueeze(0)
                        '''
                        print(f"cnmcnmcnm{k_values_pre}")
                        print(f"knmknmknm{k_values}")
                        print(f"cnmcnmcnm{cos_knowledge_subject}")
                        print(f"knmknmknm{s_values}")
                        print(f"cnmcnmcnm{cos_knowledge_relation}")
                        print(f"knmknmknm{r_values}")
                        '''
                        if ratio is None:
                            ratio = -(k_values**1) * (s_values * r_values)**1
                            ratio_sum = -k_values_pre * (s_values_pre * r_values_pre)**5
                        else:
                            ratio = torch.cat((ratio, -(k_values**1) * (s_values * r_values)**1), dim=0)
                            #print(f"cpjhhhhhcpjhhhhh{ratio.shape}")
                            ratio_sum = torch.cat((ratio_sum, -k_values_pre * (s_values_pre * r_values_pre)**5), dim=0)
                            #print(f"cpjhhhhcpjhhhh{k_values_sum.shape}")

                    v = -F.relu(-ratio_sum) #loss version 1
                    #v = -F.relu(-ratio_sum * self.lora_weight_for_training) #loss version 2
                    
                    #print((k_values_sum * cos_knowledge_subject * cos_knowledge_relation + ratio).shape)
                    if torch.max((-ratio_sum + ratio), dim=0, keepdim=True)[0] > 0:
                        DECODE_LOSS = torch.mean(-F.relu(-ratio_sum)) #loss version 1
                        #DECODE_LOSS = torch.mean(-F.relu(-ratio_sum * self.lora_weight_for_training)) #loss version 2
                    else:
                        DECODE_LOSS = torch.mean(-F.relu(-ratio_sum)) #loss version 1
                        #DECODE_LOSS = torch.mean(-F.relu(-ratio_sum * self.lora_weight_for_training)) #loss version 2
                    # / (-ratio / torch.max((-ratio), dim=0, keepdim=True)[0])
        
        return

    def set_list(self):
        #global COS_ORIGIN
        #global COS_KNOWLEDGE_1 
        #global COS_KNOWLEDGE_2 
        global COS_KNOWLEDGE_MATRIX
        #global COS_KNOWLEDGE_1_SUBJECT 
        #global COS_KNOWLEDGE_2_SUBJECT 
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        #global COS_KNOWLEDGE_1_RELATION 
        #global COS_KNOWLEDGE_2_RELATION 
        global COS_KNOWLEDGE_RELATION_MATRIX
        #COS_ORIGIN = list()
        #COS_KNOWLEDGE_1 = list()
        #COS_KNOWLEDGE_2 = list()
        COS_KNOWLEDGE_MATRIX = None
        #COS_KNOWLEDGE_1_SUBJECT = list()
        #COS_KNOWLEDGE_2_SUBJECT = list()
        COS_KNOWLEDGE_SUBJECT_MATRIX = None
        #COS_KNOWLEDGE_1_RELATION = list()
        #COS_KNOWLEDGE_2_RELATION = list()
        COS_KNOWLEDGE_RELATION_MATRIX = None

    def set_attention(self):
        global ATTENTION_MASK
        ATTENTION_MASK = None

    def get_attention(self):
        global ATTENTION_MASK
        return ATTENTION_MASK
    
    def set_decode_training(self, de):
        global DECODE_TRAINING
        DECODE_TRAINING = de

    def get_W_Norm(self):
        global W
        global NORM
        global NORM_W
        return W, NORM, NORM_W
    
    def set_W_Norm(self, W_, Norm_, Norm_W_, Down_proj_W_, label):
        global W
        global NORM
        global NORM_W
        global DOWN_PROJ_W
        global LABEL
        W = W_
        NORM = Norm_
        NORM_W = Norm_W_
        DOWN_PROJ_W = Down_proj_W_
        LABEL = label
        return
    
    
    def get_global(self):
        global DECODE_TRAINING
        global DECODE_LOSS
        global ANSWER_KEY_ID_LEN_LIST
        global ANSWER_KEY_ID_LEN_LIST_REAL
        global COMPARE
        global SAVE_REPRESENTATION
        global SAVE_REPRESENTATION_ANSWER
        global SAVE_REPRESENTATION_SR
        global ATTENTION_MASK
        global BEGIN_DECODE_TOK
        global NO_LORA
        global LORA_BLOCK_MAPPING
        global MEMORYLOSS
        global KEY_ID
        global SUBJECT_KEY_ID
        global RELATION_KEY_ID
        global ANSWER_KEY_ID
        global ANSWER_KEY_ID_REAL
        global ALREADY
        global X_BEFORE
        global REPRESENTATION_SUBJECT_MATRIX
        global REPRESENTATION_RELATION_MATRIX
        global REPRESENTATION_MATRIX
        global COS_KNOWLEDGE_MATRIX
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        global COS_KNOWLEDGE_RELATION_MATRIX
        global RATIO_1
        global RATIO_2
        
        return {
            "DECODE_TRAINING": DECODE_TRAINING,
            "DECODE_LOSS": DECODE_LOSS,
            "ANSWER_KEY_ID_LEN_LIST": ANSWER_KEY_ID_LEN_LIST,
            "ANSWER_KEY_ID_LEN_LIST_REAL": ANSWER_KEY_ID_LEN_LIST_REAL,
            "COMPARE": COMPARE,
            "SAVE_REPRESENTATION": SAVE_REPRESENTATION,
            "SAVE_REPRESENTATION_ANSWER": SAVE_REPRESENTATION_ANSWER,
            "SAVE_REPRESENTATION_SR": SAVE_REPRESENTATION_SR,
            "ATTENTION_MASK": ATTENTION_MASK,
            "BEGIN_DECODE_TOK": BEGIN_DECODE_TOK,
            "NO_LORA": NO_LORA,
            "LORA_BLOCK_MAPPING": LORA_BLOCK_MAPPING,
            "MEMORYLOSS": MEMORYLOSS,
            "KEY_ID": KEY_ID,
            "SUBJECT_KEY_ID": SUBJECT_KEY_ID,
            "RELATION_KEY_ID": RELATION_KEY_ID,
            "ANSWER_KEY_ID": ANSWER_KEY_ID,
            "ANSWER_KEY_ID_REAL": ANSWER_KEY_ID_REAL,
            "ALREADY": ALREADY,
            "X_BEFORE": X_BEFORE,
            "REPRESENTATION_SUBJECT_MATRIX": REPRESENTATION_SUBJECT_MATRIX,
            "REPRESENTATION_RELATION_MATRIX": REPRESENTATION_RELATION_MATRIX,
            "REPRESENTATION_MATRIX": REPRESENTATION_MATRIX,
            "COS_KNOWLEDGE_MATRIX": COS_KNOWLEDGE_MATRIX,
            "COS_KNOWLEDGE_SUBJECT_MATRIX": COS_KNOWLEDGE_SUBJECT_MATRIX,
            "COS_KNOWLEDGE_RELATION_MATRIX": COS_KNOWLEDGE_RELATION_MATRIX,
            "RATIO_1": RATIO_1,
            "RATIO_2": RATIO_2
        }

    def set_global(self, global_dict):
        global DECODE_TRAINING
        global DECODE_LOSS
        global ANSWER_KEY_ID_LEN_LIST
        global ANSWER_KEY_ID_LEN_LIST_REAL
        global COMPARE
        global SAVE_REPRESENTATION
        global SAVE_REPRESENTATION_ANSWER
        global SAVE_REPRESENTATION_SR
        global ATTENTION_MASK
        global BEGIN_DECODE_TOK
        global NO_LORA
        global LORA_BLOCK_MAPPING
        global MEMORYLOSS
        global KEY_ID
        global SUBJECT_KEY_ID
        global RELATION_KEY_ID
        global ANSWER_KEY_ID
        global ANSWER_KEY_ID_REAL
        global ALREADY
        global X_BEFORE
        global REPRESENTATION_SUBJECT_MATRIX
        global REPRESENTATION_RELATION_MATRIX
        global REPRESENTATION_MATRIX
        global COS_KNOWLEDGE_MATRIX
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        global COS_KNOWLEDGE_RELATION_MATRIX
        global RATIO_1
        global RATIO_2

        DECODE_TRAINING = global_dict["DECODE_TRAINING"]
        DECODE_LOSS = global_dict["DECODE_LOSS"]
        ANSWER_KEY_ID_LEN_LIST = global_dict["ANSWER_KEY_ID_LEN_LIST"]
        ANSWER_KEY_ID_LEN_LIST_REAL = global_dict["ANSWER_KEY_ID_LEN_LIST_REAL"]
        COMPARE = global_dict["COMPARE"]
        SAVE_REPRESENTATION = global_dict["SAVE_REPRESENTATION"]
        SAVE_REPRESENTATION_ANSWER = global_dict["SAVE_REPRESENTATION_ANSWER"]
        SAVE_REPRESENTATION_SR = global_dict["SAVE_REPRESENTATION_SR"]
        ATTENTION_MASK = global_dict["ATTENTION_MASK"]
        BEGIN_DECODE_TOK = global_dict["BEGIN_DECODE_TOK"]
        NO_LORA = global_dict["NO_LORA"]
        LORA_BLOCK_MAPPING = global_dict["LORA_BLOCK_MAPPING"]
        MEMORYLOSS = global_dict["MEMORYLOSS"]
        KEY_ID = global_dict["KEY_ID"]
        SUBJECT_KEY_ID = global_dict["SUBJECT_KEY_ID"]
        RELATION_KEY_ID = global_dict["RELATION_KEY_ID"]
        ANSWER_KEY_ID = global_dict["ANSWER_KEY_ID"]
        ANSWER_KEY_ID_REAL = global_dict["ANSWER_KEY_ID_REAL"]
        ALREADY = global_dict["ALREADY"]
        X_BEFORE = global_dict["X_BEFORE"]
        #REPRESENTATION_SUBJECT_MATRIX = global_dict["REPRESENTATION_SUBJECT_MATRIX"]
        #REPRESENTATION_RELATION_MATRIX = global_dict["REPRESENTATION_RELATION_MATRIX"]
        #REPRESENTATION_MATRIX = global_dict["REPRESENTATION_MATRIX"]
        #COS_KNOWLEDGE_MATRIX = global_dict["COS_KNOWLEDGE_MATRIX"]
        #COS_KNOWLEDGE_SUBJECT_MATRIX = global_dict["COS_KNOWLEDGE_SUBJECT_MATRIX"]
        #COS_KNOWLEDGE_RELATION_MATRIX = global_dict["COS_KNOWLEDGE_RELATION_MATRIX"]
        RATIO_1 = global_dict["RATIO_1"]
        RATIO_2 = global_dict["RATIO_2"]

    def set_finish_loss(self, finish):
        global FINISH
        FINISH = finish
    
    def get_begin_decode_tok(self):
        global BEGIN_DECODE_TOK
        return BEGIN_DECODE_TOK
    
    def get_x_before(self):
        global X_BEFORE
        return X_BEFORE
    
    def set_x_before(self, x_before):
        global X_BEFORE
        X_BEFORE = x_before
        return
    
    def get_overlap(self):
        global OVER
        return OVER
    
    def get_contra(self):
        global CONTR
        return CONTR
    
    def set_compare(self, save_compare):
        global COMPARE
        COMPARE = save_compare

    def save_mask(self, save_mask):
        global SAVE_MASK
        SAVE_MASK = save_mask
    
    def save_representation(self, save_representation, success_answers_begin):
        global SAVE_REPRESENTATION
        global SUCCESS_ANSWERS_BEGIN
        SAVE_REPRESENTATION = save_representation
        SUCCESS_ANSWERS_BEGIN = success_answers_begin

    def save_representation_answer(self, save_representation_answer):
        global SAVE_REPRESENTATION_ANSWER
        SAVE_REPRESENTATION_ANSWER = save_representation_answer

    def save_representation_sr(self, save_representation_sr):
        global SAVE_REPRESENTATION_SR
        SAVE_REPRESENTATION_SR = save_representation_sr
    
    def set_representation(self):
        #global REPRESENTATION_1
        #global REPRESENTATION_2
        global REPRESENTATION_MATRIX
        #global REPRESENTATION_1_SUBJECT
        #global REPRESENTATION_2_SUBJECT
        global REPRESENTATION_SUBJECT_MATRIX
        #global REPRESENTATION_1_RELATION
        #global REPRESENTATION_2_RELATION
        global REPRESENTATION_RELATION_MATRIX
        #global REPRESENTATION_1_ORIGIN
        #global REPRESENTATION_2_ORIGIN
        #global REPRESENTATION_1_ANOTHER
        #global REPRESENTATION_2_ANOTHER
        #REPRESENTATION_1 = None
        #REPRESENTATION_2 = None
        REPRESENTATION_MATRIX = None
        #REPRESENTATION_1_SUBJECT = None
        #REPRESENTATION_2_SUBJECT = None
        REPRESENTATION_SUBJECT_MATRIX = None
        #REPRESENTATION_1_RELATION = None
        #REPRESENTATION_2_RELATION = None
        REPRESENTATION_RELATION_MATRIX = None
        #REPRESENTATION_1_ORIGIN = None
        #REPRESENTATION_2_ORIGIN = None
        #REPRESENTATION_1_ANOTHER = None
        #REPRESENTATION_2_ANOTHER = None
    
    def add_adapter(self, adapter_name, config=None, memory=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name, memory=memory)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
    
    def add_grace(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_melo_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace_grace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)



    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            #print("lora")
            #print(lora_config.target_modules)
            #print("model")
            #print(key)
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _check_grace_layer_exists(self, grace_layer, key):
        #target_module_found = re.fullmatch(grace_layer, key)
        if isinstance(grace_layer, str):
            target_module_found = re.fullmatch(grace_layer, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in grace_layer)

        return target_module_found



    def _create_new_module(self, lora_config, adapter_name, target, key, memory):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "num_rank_per_block":lora_config.grace_config['num_rank_per_block']
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, key=key, memory=memory, **kwargs)

        return new_module

    def _create_new_grace_module(self, config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": config.fan_in_fan_out,
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target grace module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and 'torch.nn.Conv1D' are supported."
            )
        new_module = GraceLinear(adapter_name, in_features, out_features, config.grace_config, bias=bias, **kwargs)

        return new_module

    def _find_and_replace_grace(self, adapter_name):
        config = self.peft_config[adapter_name]
        is_target_module_in_base_model = False
        grace_layer = config.grace_layer
        key_list = [key for key,_ in self.model.named_modules()]

        for key in key_list:
            if not self._check_grace_layer_exists(grace_layer,key):
                continue
            print(f"Target Grace Layer is found: {key}")
            is_target_module_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                raise ValueError("Cannot set LoraLayer as GraceLayer")
            new_module = self._create_new_grace_module(config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)
        if not is_target_module_in_base_model:
            raise ValueError(
                f"Target grace modules {config.model.grace_layer} not found in the base model. "
                f"Please check the target modules and try again."
            )



    def _find_and_replace(self, adapter_name, memory):
        lora_config = self.peft_config[adapter_name]

        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                    lora_config.grace_config['num_rank_per_block']
                )
            elif isinstance(target, GraceLayer):
                raise ValueError("Cannot set GraceLayer as LoraLayer")
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, key, memory)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def disable_grace_layer(self):
        for module in self.model.modules():
            if isinstance(module, GraceLayer):
                module.disable_grace = True

    def enable_grace_layer(self):
        for module in self.model.modules():
            if isinstance(module, GraceLayer):
                module.disable_grace = False


    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config
    @staticmethod
    def _prepare_melo_config(peft_config, model_config):
        if peft_config.grace_layer is None:
                raise ValueError("Please specify `grace_layer` in `peft_config`")
        return peft_config


    @staticmethod
    def _prepare_grace_config(peft_config, model_config):
        if peft_config.grace_layer is None or peft_config.grace_config is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `grace_layer` and `grace_config` in `peft_config`")
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                else:
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight




class TpatcherModel(torch.nn.Module):
    def __init__(self, model, config, adapter_name, memory=None):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.config = self.peft_config[adapter_name]

        self.add_adapter(adapter_name, self.peft_config[adapter_name], memory=memory)
        self.add_grace(adapter_name,self.peft_config[adapter_name])
    
    def get_memory_loss(self):
        global MEMORYLOSS
        return MEMORYLOSS
    
    def set_memory_loss(self):
        global MEMORYLOSS
        MEMORYLOSS = 0
        return
    
    def add_adapter(self, adapter_name, config=None, memory=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name, memory=memory)
        mark_only_patch_as_trainable(self.model)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, 'patch')

    def add_grace(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_melo_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace_grace(adapter_name)
        mark_only_patch_as_trainable(self.model, self.peft_config[adapter_name].bias)

    def _find_and_replace_grace(self, adapter_name):
        config = self.peft_config[adapter_name]
        is_target_module_in_base_model = False
        grace_layer = config.grace_layer
        key_list = [key for key,_ in self.model.named_modules()]

        for key in key_list:
            if not self._check_grace_layer_exists(grace_layer,key):
                continue
            print(f"Target Grace Layer is found: {key}")
            is_target_module_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, PatchLayer):
                raise ValueError("Cannot set PatchLayer as GraceLayer")
            new_module = self._create_new_grace_module(config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)
        if not is_target_module_in_base_model:
            raise ValueError(
                f"Target grace modules {config.model.grace_layer} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _check_grace_layer_exists(self, grace_layer, key):
        #target_module_found = re.fullmatch(grace_layer, key)
        if isinstance(grace_layer, str):
            target_module_found = re.fullmatch(grace_layer, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in grace_layer)

        return target_module_found
    
    
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            #print("lora")
            #print(lora_config.target_modules)
            #print("model")
            #print(key)
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found
    
    def _create_new_module(self, lora_config, adapter_name, target, type, memory=None):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "type": type,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "num_rank_per_block":lora_config.grace_config['num_rank_per_block']
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = PatchLinear(adapter_name, in_features, out_features, bias=bias, memory=memory, **kwargs)

        return new_module

    def _create_new_grace_module(self, config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": config.fan_in_fan_out,
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target grace module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and 'torch.nn.Conv1D' are supported."
            )
        new_module = GraceLinear(adapter_name, in_features, out_features, config.grace_config, bias=bias, **kwargs)

        return new_module


    def _find_and_replace(self, adapter_name, memory):
        lora_config = self.peft_config[adapter_name]

        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, PatchLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                if "up_proj" in key:
                    new_module = self._create_new_module(lora_config, adapter_name, target, "up_proj", memory=memory)
                elif "gate_proj" in key:
                    new_module = self._create_new_module(lora_config, adapter_name, target, "gate_proj", memory=memory)
                elif "down_proj" in key:
                    new_module = self._create_new_module(lora_config, adapter_name, target, "down_proj")
                else:
                    raise ValueError(
                        f"Cannot create module at {key}"
                    )
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "patch" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config
    @staticmethod
    def _prepare_melo_config(peft_config, model_config):
        if peft_config.grace_layer is None:
                raise ValueError("Please specify `grace_layer` in `peft_config`")
        return peft_config


    @staticmethod
    def _prepare_grace_config(peft_config, model_config):
        if peft_config.grace_layer is None or peft_config.grace_config is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `grace_layer` and `grace_config` in `peft_config`")
        return peft_config
    
# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def mark_only_patch_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "patch" not in n:
            p.requires_grad = False

class mem_point:
    def __init__(self, key, value):
        self.key = key
        self.value = value
    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

    def get_lora_id(self):
        return self.value


class VecDB:
    def __init__(self, grace_config):
        self.config = grace_config
        self.table = []
        self.forget_num = 0
        self.conflict_num = 0
        self.forget_keys = []

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        return self.table[item]

    def add_cluster(self, new_key, new_value, new_edit_label):
        new_row = {'cluster_center':None, 'radius': None, 'key_label': None, 'points':[]}

        new_row['cluster_center'] = new_key.detach()
        new_row['radius'] = torch.tensor(self.config['init_radius'], device = new_key.device).view(1)
        new_row['key_label'] = new_edit_label
        new_row['points'].append(mem_point(new_key.detach(),new_value))

        self.table.append(new_row)

    def update_cluster(self, index, new_key, new_value):
        self.table[index]['points'].append(mem_point(new_key,new_value))
        key_list = [x.get_key() for x in self.table[index]['points']]
        new_cluster_center = sum(key_list)/len(key_list)
        self.table[index]['cluster_center'] = new_cluster_center

        dists = self.euc(key_list, new_cluster_center).view(-1,1)
        largest_distance, _ = dists.max(0)
        self.table[index]['radius'] = max(largest_distance, torch.tensor(self.config['init_radius'], device = new_key.device).view(1))


    def label_match(self, edit_label, key_label):
        edit_label = edit_label.masked_fill(edit_label == -100, 0)
        key_label = key_label.masked_fill(key_label == -100, 0)
        return torch.sum(edit_label) == torch.sum(key_label)

    def split_cluster_radii_in_half(self, nearest_cluster, smallest_distance):
        self.table[nearest_cluster]['radius'] = (smallest_distance / 2) - 1e-5
        self.table[-1]['radius'] = (smallest_distance / 2) + 1e-5

        cluster_radius = self.table[nearest_cluster]['radius']
        key_list = [x.get_key() for x in self.table[nearest_cluster]['points']]
        key_list = torch.stack(key_list,dim=0)
        cluster_center = self.table[nearest_cluster]['cluster_center']
        dists = self.euc(key_list,cluster_center).view(-1,1)
        filtered_key_list = []
        for index, dist in enumerate(dists):
            if dist <= cluster_radius:
                filtered_key_list.append(self.table[nearest_cluster]['points'][index])
            else:
                self.forget_keys.append(key_list[index])
        if len(filtered_key_list) == 0:
            filtered_key_list.append(mem_point(cluster_center,NO_LORA))
        self.table[nearest_cluster]['points'] = filtered_key_list

        self.forget_num += len(key_list) - len(filtered_key_list)
        self.conflict_num += 1




    def euc(self, batch_query, key):
        if isinstance(batch_query, list):
            batch_query = torch.stack(batch_query,dim=0)
        # Euclidean distance
        if len(key.shape) < 2:
            key = key.view(1, -1)
        return torch.cdist(batch_query,key, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    def search_database(self, batch_query):
        
        dists = []
        for x in self.table:
            dists.append(self.euc(batch_query,x['cluster_center']).view(-1,1))
        dists = torch.stack(dists).view(-1, len(batch_query))
        smallest_distance_list, nearest_cluster_list = dists.min(0)
        return smallest_distance_list,nearest_cluster_list

    def search_cluster(self, batch_query, smallest_distance_list,  nearest_cluster_list):
        lora_mapping_block = []
        for query, smallest_distance, nearest_cluster in zip(batch_query, smallest_distance_list,nearest_cluster_list):
            try:
                
                if smallest_distance > self.table[nearest_cluster]['radius']:
                    lora_mapping_block.append(NO_LORA)
                    continue
                # Valid Cluster
                key_list = [x.get_key() for x in self.table[nearest_cluster]['points']]
                key_list = torch.stack(key_list, dim=0)
                dists = self.euc(key_list,query).view(-1,1)
                _, nearest_key = dists.min(0)
                lora_mapping_block.append(self.table[nearest_cluster]['points'][nearest_key].get_value())
            except Exception as e:
                print(e)
                print(f'[smallest_distace]: {smallest_distance}')
                print(f"[nearest_cluster]: {self.table[nearest_cluster]['radius']}")

        return lora_mapping_block








class GraceLayer:
    def __init__(self, grace_config: dict, in_features: int, out_features: int, **kwargs):
        self.grace_config = grace_config
        self.batch_iter = None
        for k, v in grace_config.items():
            setattr(self, k, v)
        self.VecDB = VecDB(grace_config)
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.lora_block_mapping = []
        self.non_overlap_edit = 0
        self.disable_grace = False
        self.block_id = 0



    def search(self, batch_query):
        #print(batch_query.shape)
        smallest_distance_list, nearest_cluster_list = self.VecDB.search_database(batch_query)
        lora_block_mapping = self.VecDB.search_cluster(batch_query, smallest_distance_list, nearest_cluster_list)
        return smallest_distance_list, nearest_cluster_list, lora_block_mapping

    def current_block(self):
        return self.block_id
            
        

    def init_key_value(self, batch_query):
        #print(batch_query)
        lora_block_mapping = []
        for index, query in enumerate(batch_query):
            new_key = query.detach()
            new_eidt_label = self.edit_label[index]
            new_value = self.current_block()
            self.VecDB.add_cluster(new_key= new_key, new_value=new_value, new_edit_label= new_eidt_label)
            lora_block_mapping.append(new_value)
        self.block_id += 1
        return lora_block_mapping

    def add_cluster(self, query, label_index):
        new_key = query.detach()
        new_value = self.current_block()
        new_edit_label = self.edit_label[label_index]
        self.VecDB.add_cluster(new_key = new_key, new_value = new_value, new_edit_label=new_edit_label)

    
    def update_cluster(self, index, query, value):
         self.VecDB.update_cluster(index,query.detach(), value)

class GraceLinear(nn.Linear, GraceLayer):
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            grace_config: dict,
            fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features,**kwargs)
        GraceLayer.__init__(self,grace_config=grace_config, in_features=in_features, out_features=out_features)
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def forward(self, x: torch.Tensor):
        global LORA_BLOCK_MAPPING
        global SUBJECT_KEY_ID
        global RELATION_KEY_ID
        global LEN_VECTORDB
        
       
        
        global ALREADY
        global SAVE_MASK
        
        #print(x.shape)
        hhh = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        #print(x)
        layer_out = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        #print(layer_out.shape)
        if self.disable_grace:
            return layer_out

        '''Search Vector Database
        '''
        if not self.training and layer_out.shape[1] != 1:
            self.key_id = -1

        if not self.training and self.key_id == -1:
            key_id = layer_out.shape[1] - 1
            self.batch_query = layer_out[0:1, key_id]
            '''
            if layer_out.shape[1] != 1:
                print("heeeeeee")
                print(layer_out.shape)
            '''
            self.key_id = key_id
            hhh = True
        else:
            key_id = min(self.key_id, layer_out.shape[1] - 1)
        
        if not self.training:
            #batch_query = self.batch_query
            batch_query = layer_out[0:1, key_id]
        else:
            batch_query = layer_out[0:1, key_id]
        #print(batch_query)
        smallest_distance_list, nearest_cluster_list, lora_block_mapping = None, None, [NO_LORA] * layer_out.shape[0]
        if len(self.VecDB) != 0:
            smallest_distance_list, nearest_cluster_list, lora_block_mapping = self.search(batch_query)
        
        LEN_VECTORDB = len(self.VecDB)
        if not self.training:
            '''
            for index in range(layer_out.shape[1]):
                cos_origin = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION, dim=1))
                COS_ORIGIN.append(float(cos_origin.detach().cpu().numpy()))
                cos_knowledge_1 = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_1, dim=1))
                COS_KNOWLEDGE_1.append(float(cos_knowledge_1.detach().cpu().numpy()))
                cos_knowledge_2 = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_2, dim=1))
                COS_KNOWLEDGE_2.append(float(cos_knowledge_2.detach().cpu().numpy()))
                cos_knowledge_1_subject = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_1_SUBJECT, dim=1))
                COS_KNOWLEDGE_1_SUBJECT.append(float(cos_knowledge_1_subject.detach().cpu().numpy()))
                cos_knowledge_2_subject = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_2_SUBJECT, dim=1))
                COS_KNOWLEDGE_2_SUBJECT.append(float(cos_knowledge_2_subject.detach().cpu().numpy()))
                cos_knowledge_1_relation = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_1_RELATION, dim=1))
                COS_KNOWLEDGE_1_RELATION.append(float(cos_knowledge_1_relation.detach().cpu().numpy()))
                cos_knowledge_2_relation = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_2_RELATION, dim=1))
                COS_KNOWLEDGE_2_RELATION.append(float(cos_knowledge_2_relation.detach().cpu().numpy()))
                cos_knowledge_1_origin = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_1_ORIGIN, dim=1))
                cos_knowledge_2_origin = torch.einsum('bd,bd->b', nn.functional.normalize(layer_out[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_2_ORIGIN, dim=1))
                print(f"check the existing knowledge in already {ALREADY}: origin: {cos_origin} k1: {cos_knowledge_1} k2: {cos_knowledge_2} sub1: {cos_knowledge_1_subject} sub2: {cos_knowledge_2_subject} rel1: {cos_knowledge_1_relation} rel2: {cos_knowledge_2_relation} origin_q1: {cos_knowledge_1_origin} origin_q2: {cos_knowledge_2_origin}")
            '''

            #lora_block_mapping = [list(range(len(self.VecDB)))[1]]
            #if hhh:  
            #multi的
            '''
            if True: 
                #lora_block_mapping = [list(range(len(self.VecDB)))[1]] 
                lora_block_mapping = list(range(len(self.VecDB)))
                lora_block_mapping = list()
                if max(COS_KNOWLEDGE_1) < 0.6:
                    lora_block_mapping.append(0)
                elif max(COS_KNOWLEDGE_2) < 0.6:
                    lora_block_mapping.append(1)
                if ((max(COS_KNOWLEDGE_1) > 0.6 or max(COS_KNOWLEDGE_2) > 0.6) and cos_origin > 0.35) or len(lora_block_mapping) == 0:
                    lora_block_mapping = [-100]
            '''
            #port的
            lora_block_mapping = list(range(len(self.VecDB)))
            

            self.lora_block_mapping = LORA_BLOCK_MAPPING = lora_block_mapping
            print(self.lora_block_mapping)
            
            return layer_out
        
        '''
        if not self.training:
            self.lora_block_mapping = LORA_BLOCK_MAPPING = lora_block_mapping
            #print("hhhhhhhhh")
            #print(LORA_BLOCK_MAPPING)
            return layer_out
        '''

        '''Only update the vector database once per batch
        '''
        if len(self.VecDB) == 0:
            self.lora_block_mapping = self.init_key_value(batch_query)
        elif self.batch_iter == 0:
            self.lora_block_mapping = [self.block_id]# * layer_out.shape[0]
            
            for index, query in enumerate(batch_query):
                row = self.VecDB[nearest_cluster_list[index]]
                
                if smallest_distance_list[index] > row['radius'] + self.init_radius:
                    self.add_cluster(query,label_index=index)
                elif self.VecDB.label_match(self.edit_label[index], row['key_label']):
                    print(f'The {index}th query is close to a previous edit, the labels are the same')
                    self.add_cluster(query,label_index=index)
                    #self.update_cluster(nearest_cluster_list[index],query,self.block_id)
                else:
                    print(f'The {index}th query is close to a previous edit, but the labels are different')
                    self.add_cluster(query,label_index = index)
                    #self.VecDB.split_cluster_radii_in_half(nearest_cluster_list[index], smallest_distance_list[index])
             
            self.block_id += 1
        else:
            pass
        
        LORA_BLOCK_MAPPING = self.lora_block_mapping
        LEN_VECTORDB = len(self.VecDB)
        #print(LORA_BLOCK_MAPPING)
        '''
        REPRESENTATION = layer_out[:, key_id]
        if len(self.VecDB) == 1:
            REPRESENTATION_1_SUBJECT = layer_out[:, SUBJECT_KEY_ID]
            REPRESENTATION_1_RELATION = layer_out[:, RELATION_KEY_ID]
            REPRESENTATION_1_ORIGIN = layer_out[:, key_id]
            REPRESENTATION_1 = layer_out[:, -1]
        elif len(self.VecDB) == 2:
            REPRESENTATION_2_SUBJECT = layer_out[:, SUBJECT_KEY_ID]
            REPRESENTATION_2_RELATION = layer_out[:, RELATION_KEY_ID]
            REPRESENTATION_2_ORIGIN = layer_out[:, key_id]
            REPRESENTATION_2 = layer_out[:, -1]

        COS_ORIGIN = list()
        COS_KNOWLEDGE_1 = list()
        COS_KNOWLEDGE_2 = list()
        COS_KNOWLEDGE_1_SUBJECT = list()
        COS_KNOWLEDGE_2_SUBJECT = list()
        COS_KNOWLEDGE_1_RELATION = list()
        COS_KNOWLEDGE_2_RELATION = list()
        '''

        return layer_out

class dynamic(nn.Module):
    def __init__(
            self,
            maximum_rank: int = 1,
            num_rank_per_block: int = 1
    ):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        super(dynamic, self).__init__()
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0
    def get_block_dimension(self):
        return self.maximum_block

    def get_block(self):
        return self.current_block

    def set_block(self, block):
        self.current_block = max(0, min(block, self.get_block()))

    def block_rank_mapping(self, block_id):
        start = block_id * self.num_rank_per_block
        end = start + self.num_rank_per_block
        return start, end

    def update_dynamic(self, maximum_rank, num_rank_per_block):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0

    def forward(self, inputs):
        #print(inputs.shape)
        block_list = []
        assert len(LORA_BLOCK_MAPPING) != 0, "No element in LORA_BLOCK_MAPPING"
        #print("lora mapping {}".format(LORA_BLOCK_MAPPING))
        for block_id in LORA_BLOCK_MAPPING:
            if block_id == NO_LORA:
                zero_tensor = torch.zeros((self.num_rank_per_block,inputs.shape[1]),device=inputs.device)
                block_list.append(zero_tensor)
            else:
                start, end = self.block_rank_mapping(block_id)
                #print(start)
                #print(end)
                #print(inputs.shape)
                block_list.append(inputs[start:end])
        #result = torch.stack(block_list,1)

        result = torch.cat(block_list,dim=0).unsqueeze(0)
        #print(block_list[0].shape)
        #print(result.shape)

        return result * math.sqrt(self.maximum_rank / self.num_rank_per_block)


class patchdynamic(nn.Module):
    def __init__(
            self,
            type: str = "up_proj",
            maximum_rank: int = 1,
            num_rank_per_block: int = 1
    ):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        super(patchdynamic, self).__init__()
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0
        self.type = type
        #print("cnmcnmcnm:{}".format(self.type))
    def get_block_dimension(self):
        return self.maximum_block

    def get_block(self):
        return self.current_block

    def set_block(self, block):
        self.current_block = max(0, min(block, self.get_block()))

    def block_rank_mapping(self, block_id):
        start = block_id * self.num_rank_per_block
        end = start + self.num_rank_per_block
        return start, end

    def update_dynamic(self, maximum_rank, num_rank_per_block):
        assert maximum_rank % num_rank_per_block == 0, \
            "Maximum_rank % num_rank_per_block == 0 should be True"
        self.maximum_rank = maximum_rank
        self.num_rank_per_block = num_rank_per_block
        self.maximum_block = maximum_rank // num_rank_per_block
        self.current_block = 0

    def forward(self, weight, bias):
        #print(inputs.shape)
        weight = weight
        #print("get:{}".format(weight.device))
        bias = bias
        block_weight_list = []
        block_bias_list = []
        assert len(LORA_BLOCK_MAPPING) != 0, "No element in LORA_BLOCK_MAPPING"
        #print("lora mapping {}".format(LORA_BLOCK_MAPPING))
        for block_id in LORA_BLOCK_MAPPING:
            if block_id == NO_LORA:
                zero_weight_tensor = torch.zeros((self.num_rank_per_block,weight.shape[1]),device=weight.device)
                zero_bias_tensor = torch.zeros((self.num_rank_per_block),device=bias.device)
                block_weight_list.append(zero_weight_tensor)
                block_bias_list.append(zero_bias_tensor)
            else:
                start, end = self.block_rank_mapping(block_id)
                #print(start)
                #print(end)
                #print(inputs.shape)
                block_weight_list.append(weight[start:end])
                block_bias_list.append(bias[start:end])
        #result = torch.stack(block_list,1)
        result_weight = torch.cat(block_weight_list,dim=0)#.unsqueeze(0)
        result_bias = torch.cat(block_bias_list,dim=0)#.unsqueeze(0)
        #print(block_list[0].shape)
        #print(result.shape)

        return result_weight * math.sqrt(self.maximum_rank / self.num_rank_per_block), result_bias * math.sqrt(self.maximum_rank / self.num_rank_per_block)



class LoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.nd_lora_A = dynamic()
        self.nd_lora_B = dynamic()


        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_rank_per_block):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.in_features, r)))})
            self.lora_B = nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.out_features)))})
            self.nd_lora_A.update_dynamic(r, num_rank_per_block)
            self.nd_lora_B.update_dynamic(r,num_rank_per_block)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            #nn.init.orthogonal_(self.lora_A[adapter_name])
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])
            print(self.lora_A[adapter_name].shape)
            '''
            #initialize version 2
            with torch.no_grad():
                i = 0
                while i < self.lora_A[adapter_name].shape[1]:
                    self.lora_A[adapter_name][:, i:i+2] = self.lora_A[adapter_name][:, 0:2].clone()
                    i = i + 2
            '''
            '''
            #initialize version 3
            nn.init.orthogonal_(self.lora_A[adapter_name])
            '''
            

        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])



class PatchLayer:
    def __init__(self, in_features: int, out_features: int, type: str, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.patch_dropout = nn.ModuleDict({})
        self.patch = nn.ModuleDict({})
        self.nd_patch = patchdynamic(type)
        self.in_features = in_features
        self.out_features = out_features
        self.type = type
        self.kwargs = kwargs
        self.memory = list()
    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, num_rank_per_block):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Identity()
            #lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        
        self.patch_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            if self.type == "up_proj" or self.type == "gate_proj":
                self.patch = nn.ParameterDict({adapter_name: nn.Linear(self.in_features, r)})
            elif self.type == "down_proj":
                self.patch = nn.ParameterDict({adapter_name: nn.Linear(r, self.out_features)})
            self.nd_patch.update_dynamic(r, num_rank_per_block)
            self.scaling[adapter_name] = lora_alpha / r
        self.to(self.weight.device)
        #print("send:{}".format(self.weight.device))
        #print("send:{}".format(self.patch['default'].weight.device))


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_rank_per_block: int = 1,
        key: str = "",
        memory=None,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.key = key
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_rank_per_block)
        self.lora_block_mapping = LORA_BLOCK_MAPPING
        self.active_adapter = adapter_name
        self.KLDiv = torch.nn.KLDivLoss(reduction="none")
        self.lora_weight = None
        self.previous_mask_answer = None
        self.mask_answer_list = None 
        self.extra_weight = None
        self.memory = memory

    def update_stable_tok(self):
        if self.mask_answer_list is None:
            self.mask_answer_list = self.previous_mask_answer
        else:
            self.mask_answer_list = torch.cat((self.mask_answer_list, self.previous_mask_answer), dim=0)

    def get_stable_tok(self):
        global ANSWER_KEY_ID_LEN_LIST_REAL
        

        lora_weight = self.lora_weight.weight.clone()   
        if self.mask_answer_list is not None:        
            lora_weight[:, self.mask_answer_list] = -float('inf')
                        
        _, lora_index = torch.topk(lora_weight[0], 1, largest=True, sorted=True)

        self.previous_mask_answer = lora_index

        
        return ANSWER_KEY_ID_LEN_LIST_REAL[lora_index]
    
    def set_stable_tok(self):
        self.previous_mask_answer = None
        self.mask_answer_list = None

    def initialize_lora_weights(self,):
        global COS_KNOWLEDGE_MATRIX
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        global COS_KNOWLEDGE_RELATION_MATRIX
        global COS_KNOWLEDGE_MATRIX_COPY
        global COS_KNOWLEDGE_SUBJECT_MATRIX_COPY
        global COS_KNOWLEDGE_RELATION_MATRIX_COPY
        global ANSWER_KEY_ID_LEN_LIST
        global CAL_PROB
        if CAL_PROB:
            self.lora_weight = nn.Linear(COS_KNOWLEDGE_MATRIX_COPY.shape[1], 1, bias=False, device=self.weight.device)
            
            ratio = None
            ratio_for_index = None
            for i in range(len(ANSWER_KEY_ID_LEN_LIST)):
                '''
                print(i)
                print(ANSWER_KEY_ID_LEN_LIST[i])
                print(COS_KNOWLEDGE_MATRIX[:, i].shape)
                '''
                k_values, _ = torch.topk(COS_KNOWLEDGE_MATRIX_COPY[:, i], min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX_COPY[:, i].shape[0]), largest=True, sorted=True)
                k_values = k_values.mean().unsqueeze(0)
                
                s_values = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX_COPY[:, i], dim=0, keepdim=True)[0]
                s_values = s_values.mean().unsqueeze(0) 
                r_values = torch.max(COS_KNOWLEDGE_RELATION_MATRIX_COPY[:, i], dim=0, keepdim=True)[0]
                r_values = r_values.mean().unsqueeze(0)
                '''
                print("rrrrrrrrrrrrrrrrrrrrrrrrr")
                print(r_values)
                print("sssssssssssssssssssssssss")
                print(s_values)
                print("kkkkkkkkkkkkkkkkkkkkkkkkk")
                print(k_values)
                '''
                if ratio is None:
                    #ratio = -k_values * s_values * r_values
                    ratio = s_values
                    ratio_for_index = s_values
                else:
                    #ratio = torch.cat((ratio, -k_values * s_values * r_values), dim=0)
                    ratio = torch.cat((ratio, s_values), dim=0)
                    ratio_for_index = torch.cat((ratio_for_index, s_values), dim=0)


            self.lora_weight.weight.data = torch.ones(1, COS_KNOWLEDGE_MATRIX_COPY.shape[1], device=ratio.device)
            self.lora_weight_for_training = ratio.reshape(1, COS_KNOWLEDGE_MATRIX_COPY.shape[1])
            self.lora_weight_for_index = ratio_for_index.reshape(1, COS_KNOWLEDGE_MATRIX_COPY.shape[1])
        else:
            self.lora_weight = nn.Linear(COS_KNOWLEDGE_MATRIX.shape[1], 1, bias=False, device=self.weight.device)
            
            ratio = None
            ratio_for_index = None
            for i in range(len(ANSWER_KEY_ID_LEN_LIST)):
                '''
                print(i)
                print(ANSWER_KEY_ID_LEN_LIST[i])
                print(COS_KNOWLEDGE_MATRIX[:, i].shape)
                '''
                k_values, _ = torch.topk(COS_KNOWLEDGE_MATRIX[:, i], min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX[:, i].shape[0]), largest=True, sorted=True)
                k_values = k_values.mean().unsqueeze(0)
                
                s_values = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX[:, i], dim=0, keepdim=True)[0]
                s_values = s_values.mean().unsqueeze(0) 
                r_values = torch.max(COS_KNOWLEDGE_RELATION_MATRIX[:, i], dim=0, keepdim=True)[0]
                r_values = r_values.mean().unsqueeze(0)
                '''
                print("rrrrrrrrrrrrrrrrrrrrrrrrr")
                print(r_values)
                print("sssssssssssssssssssssssss")
                print(s_values)
                print("kkkkkkkkkkkkkkkkkkkkkkkkk")
                print(k_values)
                '''
                if ratio is None:
                    #ratio = -k_values * s_values * r_values
                    ratio = (s_values)
                    ratio_for_index = s_values
                else:
                    #ratio = torch.cat((ratio, -k_values * s_values * r_values), dim=0)
                    ratio = torch.cat((ratio, (s_values)), dim=0)
                    ratio_for_index = torch.cat((ratio_for_index, s_values), dim=0)


            self.lora_weight.weight.data = torch.ones(1, COS_KNOWLEDGE_MATRIX.shape[1], device=ratio.device)
            self.lora_weight_for_training = ratio.reshape(1, COS_KNOWLEDGE_MATRIX.shape[1])
            self.lora_weight_for_index = ratio_for_index.reshape(1, COS_KNOWLEDGE_MATRIX.shape[1])

        


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def _nor_loss(self, logits, dg_logits, tau=3):
        pred_probs = F.log_softmax(logits / tau, dim=0)
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / tau, dim=0)
        loss = (tau ** 2) * self.KLDiv(pred_probs, dg_probs)
        loss = torch.mean(loss)

        return loss

    def forward(self, x: torch.Tensor):
        rank = 1
        global LORA_BLOCK_MAPPING
        global LAST_INPUT_TOKEN
        global I
        global ONLY_ONE_QUESTION
        global BEAM
        global LEN_VECTORDB
        global extra_weight
        global COMPARE
        global MEMORYLOSS 
        global DECODE_LOSS
        global FINISH_LOSS_POS
        global FINISH_LOSS_NEG
        global FINISH_LOSS_HIDDEN
        global ANSWER_KEY_ID_LEN_LIST_REAL
        global ANSWER_KEY_ID_LEN_LIST0
        global SAVE_MASK
        global SAVE_REPRESENTATION
        global SAVE_REPRESENTATION_ANSWER
        global SAVE_REPRESENTATION_SR
        global ATTENTION_MASK
        global BEGIN_DECODE_TOK
        global ALREADY
        global KEY_ID
        global ANSWER_KEY_ID_REAL
        global ANSWER_KEY_ID
        global SUBJECT_KEY_ID
        global RELATION_KEY_ID
        global W
        global NORM
        global NORM_W
        global DOWN_PROJ_W
        global LABEL
        global X_BEFORE
        global MASK_MATRIX
        global MASK_MATRIX_REAL
        global MASK_MATRIX_REAL_INCLUDEINPUT
        #global REPRESENTATION
        #global REPRESENTATION_1
        #global REPRESENTATION_2
        global REPRESENTATION_MATRIX
        #global REPRESENTATION_1_SUBJECT
        #global REPRESENTATION_2_SUBJECT
        global REPRESENTATION_SUBJECT_MATRIX
        #global REPRESENTATION_1_RELATION
        #global REPRESENTATION_2_RELATION
        global REPRESENTATION_RELATION_MATRIX
        #global COS_KNOWLEDGE_1
        #global COS_KNOWLEDGE_2
        global COS_KNOWLEDGE_MATRIX
        #global COS_KNOWLEDGE_1_SUBJECT
        #global COS_KNOWLEDGE_2_SUBJECT
        global COS_KNOWLEDGE_SUBJECT_MATRIX
        #global COS_KNOWLEDGE_1_RELATION
        #global COS_KNOWLEDGE_2_RELATION
        global COS_KNOWLEDGE_RELATION_MATRIX
        #global RATIO_1
        #global RATIO_2
        #global REPRESENTATION_1_ANOTHER
        #global REPRESENTATION_2_ANOTHER
        #global REPRESENTATION_1_ANOTHER_ORIGIN
        #global REPRESENTATION_2_ANOTHER_ORIGIN
        global DECODE_TRAINING
        global CAL_PROB
        global SUCCESS_ANSWERS_BEGIN
        global NUM_SUCCESS_ANSWERS
        global FINISH
        global SEARCH
        global MAX_INDEX
        global MAX_INDEX_WEIGHT 
        global OVER
        global CONTR

        if BEAM:
            self.lora_weight = nn.Linear(len(ANSWER_KEY_ID_LEN_LIST), 1, bias=False, device=self.weight.device)
            self.lora_weight_for_training = torch.ones(1, len(ANSWER_KEY_ID_LEN_LIST), device=self.weight.device)
            self.lora_weight_for_index = torch.ones(1, len(ANSWER_KEY_ID_LEN_LIST), device=self.weight.device)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            lora_A = self.nd_lora_A(self.lora_A[self.active_adapter].T).mT
            lora_B = self.nd_lora_B(self.lora_B[self.active_adapter])
            #print(f'phmphmcnmcnmcnmphmphm:{lora_A.shape} {lora_B.shape}')
            #print(x.shape)
            #print((self.lora_dropout[self.active_adapter](x) @ lora_A).shape)
            #result += (self.lora_dropout[self.active_adapter](x) @ lora_A @ lora_B) \
                      #* self.scaling[self.active_adapter]
            
            
            
            mask = torch.zeros((result.shape[1], result.shape[1])).to(result.device)
            if result.shape[1] > 1:
                if lora_A.shape[2] > rank and not SAVE_REPRESENTATION:
                    for key in range(result.shape[1]-1, result.shape[1]):
                        mask[key][key] = 0
                else:
                    for key in range(KEY_ID, result.shape[1]):
                        mask[key][key] = 1

                if BEAM:
                    for key in range(result.shape[1]-1, result.shape[1]):
                        mask[key][key] = 1
                
                ALREADY = 0

                #TODO: 决定哪个参数更重要
                '''
                if lora_A.shape[2] > 2 and X_BEFORE is not None:
                    x_copy = x.detach().clone()
                    origin = result.detach().clone()
                    #print(lora_A.shape)
                    #print(lora_B.shape)
                    after_1 = origin + torch.einsum('km,bkd->bmd', mask, (self.lora_dropout[self.active_adapter](x_copy) @ lora_A[:, :, 0:2] @ lora_B[:, 0:2, :]) * self.scaling[self.active_adapter])
                    after_2 = origin + torch.einsum('km,bkd->bmd', mask, (self.lora_dropout[self.active_adapter](x_copy) @ lora_A[:, :, 2:4] @ lora_B[:, 2:4, :]) * self.scaling[self.active_adapter])
                    #print(X_BEFORE.shape)
                    distribution_x = torch.softmax(torch.matmul(W, NORM(X_BEFORE.to(NORM_W.device))[0][-1].to(W.device)), dim=0)
                    distribution_origin = torch.matmul(W, NORM(origin.to(NORM_W.device))[0][-1].to(W.device))
                    distribution_after_1 = torch.matmul(W, NORM(after_1.to(NORM_W.device))[0][-1].to(W.device))
                    distribution_after_2 = torch.matmul(W, NORM(after_2.to(NORM_W.device))[0][-1].to(W.device))
                    distribution_change_origin = self._nor_loss(distribution_origin, distribution_x)
                    distribution_change_after_1 = self._nor_loss(distribution_after_1, distribution_x)
                    distribution_change_after_2 = self._nor_loss(distribution_after_2, distribution_x)
                    distribution_change_origin_after_1 = self._nor_loss(distribution_after_1, distribution_origin)
                    distribution_change_origin_after_2 = self._nor_loss(distribution_after_2, distribution_origin)
                    
                    print(f"check the mutual information in already {ALREADY}: origin: {distribution_change_origin} after1: {distribution_change_after_1} after2: {distribution_change_after_2} oa1: {distribution_change_origin_after_1} oa2: {distribution_change_origin_after_2}")
                    X_BEFORE = None
                '''
                #multi的
                '''
                if lora_A.shape[2] > 2:
                    lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                '''
                

                #port的
                '''
                if lora_A.shape[2] > 2:
                    print(f"max sub1 of {ALREADY}: {max(COS_KNOWLEDGE_1_SUBJECT)}")
                    print(f"max sub2 of {ALREADY}: {max(COS_KNOWLEDGE_2_SUBJECT)}")
                    print(f"max rel1 of {ALREADY}: {max(COS_KNOWLEDGE_1_RELATION)}")
                    print(f"max rel2 of {ALREADY}: {max(COS_KNOWLEDGE_2_RELATION)}")
                    print(f"max ans1 of {ALREADY}: {max(COS_KNOWLEDGE_1)}")
                    print(f"max ans2 of {ALREADY}: {max(COS_KNOWLEDGE_2)}")
                    print(f"ratio_wa 1 of {ALREADY}: {(max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION))**20}")
                    print(f"ratio_wa 2 of {ALREADY}: {(max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION))**20}")
                    print(f"ratio 1 of {ALREADY}: {max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)}")
                    print(f"ratio 2 of {ALREADY}: {max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)}")
                    RATIO_1 = max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)
                    RATIO_2 = max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)
                    if RATIO_1 > (RATIO_2 / 1):
                        if (max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) < 1.1:
                            lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * 0
                            lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * ((max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) / RATIO_2)
                        else:
                            lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * ((max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) / RATIO_1)
                            lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                        if (max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) < 1.1:
                            lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                    else:
                        if (max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) < 1.1:
                            lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                            lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * ((max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) / RATIO_1)
                        else:
                            lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * ((max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) / RATIO_2)
                            lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * 0
                        if (max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) < 1.1:
                            lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * 0
                '''
                '''
                if self.key == "model.layers.23.mlp.down_proj":
                    lora_A[:, :, :] = lora_A[:, :, :] * 0               
                if lora_A.shape[2] > 2:
                    if self.key == "model.layers.15.mlp.down_proj":
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                    if self.key == "model.layers.24.mlp.down_proj":
                        lora_A[:, :, 0:4] = lora_A[:, :, 0:4] * 0

                    if self.key == "model.layers.23.mlp.down_proj":
                        cos_knowledge_1 = torch.einsum('bd,bd->b', nn.functional.normalize(result[:, -1], dim=1), nn.functional.normalize(REPRESENTATION_1_ANOTHER, dim=1))
                        #COS_KNOWLEDGE_1.append(cos_knowledge_1)
                        cos_knowledge_2 = torch.einsum('bd,bd->b', nn.functional.normalize(result[:, -1], dim=1), nn.functional.normalize(REPRESENTATION_2_ANOTHER, dim=1))
                        #COS_KNOWLEDGE_2.append(cos_knowledge_2)
                        cos_knowledge_1_origin = torch.einsum('bd,bd->b', nn.functional.normalize(result[:, -1], dim=1), nn.functional.normalize(REPRESENTATION_1_ANOTHER_ORIGIN, dim=1))
                        cos_knowledge_2_origin = torch.einsum('bd,bd->b', nn.functional.normalize(result[:, -1], dim=1), nn.functional.normalize(REPRESENTATION_2_ANOTHER_ORIGIN, dim=1))
                        print(f"check the existing knowledge in 23 in already {ALREADY}: k1: {cos_knowledge_1} k2: {cos_knowledge_2} origin_q1: {cos_knowledge_1_origin} origin_q2: {cos_knowledge_2_origin}")
                else:
                    if self.key == "model.layers.23.mlp.down_proj":
                        if REPRESENTATION_1 is not None and REPRESENTATION_2 is None:
                            REPRESENTATION_1_ANOTHER_ORIGIN = result[:, KEY_ID]
                            REPRESENTATION_1_ANOTHER = result[:, -1]
                        elif REPRESENTATION_1 is not None and REPRESENTATION_2 is not None:
                            REPRESENTATION_2_ANOTHER_ORIGIN = result[:, KEY_ID]
                            REPRESENTATION_2_ANOTHER = result[:, -1]
                    if self.key == "model.layers.15.mlp.down_proj":
                        if REPRESENTATION_1 is not None and REPRESENTATION_2 is not None:
                            #print(f"nmbnmbnmb:{lora_A.shape}")
                            lora_A[:, :, :] = lora_A[:, :, :] * 0
                    if self.key == "model.layers.24.mlp.down_proj":
                        if REPRESENTATION_1 is not None and REPRESENTATION_2 is None:
                            lora_A[:, :, :] = lora_A[:, :, :] * 0
                '''
                
            else:
                mask[0][0] = 1
                ALREADY = ALREADY + 1
                '''
                print(f"max sub1 of {ALREADY}: {max(COS_KNOWLEDGE_1_SUBJECT)}")
                print(f"max sub2 of {ALREADY}: {max(COS_KNOWLEDGE_2_SUBJECT)}")
                print(f"max rel1 of {ALREADY}: {max(COS_KNOWLEDGE_1_RELATION)}")
                print(f"max rel2 of {ALREADY}: {max(COS_KNOWLEDGE_2_RELATION)}")
                print(f"max ans1 of {ALREADY}: {max(COS_KNOWLEDGE_1)}")
                print(f"max ans2 of {ALREADY}: {max(COS_KNOWLEDGE_2)}")
                print(f"ratio_wa 1 of {ALREADY}: {(max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION))**20}")
                print(f"ratio_wa 2 of {ALREADY}: {(max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION))**20}")
                print(f"ratio 1 of {ALREADY}: {max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)}")
                print(f"ratio 2 of {ALREADY}: {max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)}")
                if RATIO_1 > (RATIO_2 / 1):
                    if (max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) < 1.1:
                        lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * 0
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * ((max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) / RATIO_2)
                    else:
                        lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * ((max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) / RATIO_1)
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                    if (max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) < 1.1:
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                else:
                    if (max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) < 1.1:
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                        lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * ((max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) / RATIO_1)
                    else:
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * ((max(COS_KNOWLEDGE_2_SUBJECT) * max(COS_KNOWLEDGE_2_RELATION) / max(COS_KNOWLEDGE_2)) / RATIO_2)
                        lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * 0
                    if (max(COS_KNOWLEDGE_1_SUBJECT) * max(COS_KNOWLEDGE_1_RELATION) / max(COS_KNOWLEDGE_1)) < 1.1:
                        lora_A[:, :, 0:2] = lora_A[:, :, 0:2] * 0

                if max(COS_KNOWLEDGE_1_SUBJECT) != max(COS_KNOWLEDGE_1_SUBJECT[0:-1]) or max(COS_KNOWLEDGE_2_SUBJECT) != max(COS_KNOWLEDGE_2_SUBJECT[0:-1]):
                    #print(max(COS_KNOWLEDGE_1_SUBJECT))
                    #print(max(COS_KNOWLEDGE_1_SUBJECT[0:-1]))
                    #print(ALREADY)
                    #print("cnmcnmcnmcnmcnmcnmcnmcnmcnmcnmcnmcnm")
                    ATTENTION_MASK = sorted([COS_KNOWLEDGE_1_SUBJECT.index(max(COS_KNOWLEDGE_1_SUBJECT)), COS_KNOWLEDGE_1_RELATION.index(max(COS_KNOWLEDGE_1_RELATION))])
                    BEGIN_DECODE_TOK = ALREADY + 1
                '''
                #TODO: 决定哪个参数更重要
                '''
                if lora_A.shape[2] > 2 and X_BEFORE is not None:
                    x_copy = x.detach().clone()
                    origin = result.detach().clone()
                    #print(lora_A.shape)
                    #print(lora_B.shape)
                    after_1 = origin + torch.einsum('km,bkd->bmd', mask, (self.lora_dropout[self.active_adapter](x_copy) @ lora_A[:, :, 0:2] @ lora_B[:, 0:2, :]) * self.scaling[self.active_adapter])
                    after_2 = origin + torch.einsum('km,bkd->bmd', mask, (self.lora_dropout[self.active_adapter](x_copy) @ lora_A[:, :, 2:4] @ lora_B[:, 2:4, :]) * self.scaling[self.active_adapter])
                    #print(X_BEFORE.shape)
                    distribution_x = torch.softmax(torch.matmul(W, NORM(X_BEFORE.to(NORM_W.device))[0][-1].to(W.device)), dim=0)
                    distribution_origin = torch.matmul(W, NORM(origin.to(NORM_W.device))[0][-1].to(W.device))
                    distribution_after_1 = torch.matmul(W, NORM(after_1.to(NORM_W.device))[0][-1].to(W.device))
                    distribution_after_2 = torch.matmul(W, NORM(after_2.to(NORM_W.device))[0][-1].to(W.device))
                    distribution_change_origin = self._nor_loss(distribution_origin, distribution_x)
                    distribution_change_after_1 = self._nor_loss(distribution_after_1, distribution_x)
                    distribution_change_after_2 = self._nor_loss(distribution_after_2, distribution_x)
                    distribution_change_origin_after_1 = self._nor_loss(distribution_after_1, distribution_origin)
                    distribution_change_origin_after_2 = self._nor_loss(distribution_after_2, distribution_origin)
                    print(f"check the mutual information in already {ALREADY}: origin: {distribution_change_origin} after1: {distribution_change_after_1} after2: {distribution_change_after_2} oa1: {distribution_change_origin_after_1} oa2: {distribution_change_origin_after_2}")
                    X_BEFORE = None
                '''
                
                #answer_position_list = [4, 6] #id=0样本,insert位置329,第一个answer长度1,第二个answer长度3
                #answer_position_list = [7, 11] #id=7样本,insert位置330,第一个answer长度6,第二个answer长度3
                '''
                answer_position_list = [8, 17] #id=10样本,insert位置332,第一个answer长度7,第二个answer长度8
                answer_scale_list = [0.0, 0.0]
                answer_scale_list_post = [0.0, 0.0]
                
                for pos in range(len(answer_position_list)):
                    
                    if ALREADY >= answer_position_list[pos]:
                        lora_A[:, :, 2*pos:2*(pos+1)] = lora_A[:, :, 2*pos:2*(pos+1)] * answer_scale_list[pos]
                    if pos >= 1 and ALREADY < answer_position_list[pos-1]:
                        lora_A[:, :, 2*pos:2*(pos+1)] = lora_A[:, :, 2*pos:2*(pos+1)] * answer_scale_list_post[pos]
                '''                     
                '''
                if ALREADY == 3:
                    lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                else:
                    lora_A[:, :, 0:4] = lora_A[:, :, 0:4] * 0
                '''
                #port的
                '''
                print(self.key)
                if self.key == "model.layers.23.mlp.down_proj":
                    lora_A[:, :, :] = lora_A[:, :, :] * 0               
                if lora_A.shape[2] > 2:
                    if self.key == "model.layers.15.mlp.down_proj":
                        lora_A[:, :, 2:4] = lora_A[:, :, 2:4] * 0
                    if self.key == "model.layers.24.mlp.down_proj":
                        lora_A[:, :, 0:4] = lora_A[:, :, 0:4] * 0

                    if self.key == "model.layers.23.mlp.down_proj":
                        cos_knowledge_1 = torch.einsum('bd,bd->b', nn.functional.normalize(result[:, -1], dim=1), nn.functional.normalize(REPRESENTATION_1_ANOTHER, dim=1))
                        #COS_KNOWLEDGE_1.append(cos_knowledge_1)
                        cos_knowledge_2 = torch.einsum('bd,bd->b', nn.functional.normalize(result[:, -1], dim=1), nn.functional.normalize(REPRESENTATION_2_ANOTHER, dim=1))
                        #COS_KNOWLEDGE_2.append(cos_knowledge_2)
                        print(f"check the existing knowledge in 23 in already {ALREADY}: k1: {cos_knowledge_1} k2: {cos_knowledge_2}")
                else:
                    pass
                '''
                
            #print(result[0][-1])
            #print(f'phmphmcnmcnmcnmphmphm:{((self.lora_dropout[self.active_adapter](x) @ lora_A @ lora_B) * self.scaling[self.active_adapter]).shape}')
            #print(((self.lora_dropout[self.active_adapter](x) @ lora_A @ lora_B) \
                      #* self.scaling[self.active_adapter])[0][-1])
            
            #lora_A = torch.ones_like(lora_A, device=lora_A.device)
            #连续版
            if SAVE_REPRESENTATION_SR or COMPARE:
                lora_A[:, :, :] = lora_A[:, :, :] * 0
            '''
            if self.mask_answer_list is not None:
                lora_A[:, :, 2*self.mask_answer_list] = lora_A[:, :, 2*self.mask_answer_list] * 0 
                lora_A[:, :, 2*self.mask_answer_list+1] = lora_A[:, :, 2*self.mask_answer_list+1] * 0 
            '''
             
            if lora_A.shape[2] > rank:
                if SAVE_REPRESENTATION:
                    
                    lora_A[:, :, 0:-rank] = lora_A[:, :, 0:-rank] * 0
                else:
                    
                    if COMPARE or (not BEAM and result.shape[1] > 1):
                        lora_A[:, :, :] = lora_A[:, :, :] * 0
                    else:
                        
                        #TODO: 这里是softmax+topk取
                        '''
                        real_lora_weight = F.softmax(self.lora_weight_for_index.to(lora_A.device), dim=1)
                        _, lora_index = torch.topk(real_lora_weight[0], 10, largest=True, sorted=True)
                        '''
                        #TODO: 过滤机制
                        #TODO: 这里是按阈值取
                        
                        real_lora_weight = self.lora_weight_for_index.to(lora_A.device)
                        '''
                        if not CAL_PROB:
                            print("zzzzzccccc")
                            print(real_lora_weight)
                        '''
                        '''
                        alpha = -10000
                        lora_index = list()
                        for r_l_w in range(len(real_lora_weight[0])):
                            if real_lora_weight[0][r_l_w] > alpha:
                               lora_index.append(r_l_w) 
                        lora_index = torch.tensor(lora_index, device=lora_A.device)
                        '''
                        lora_index = torch.tensor(list(range(len(real_lora_weight[0]))), device=lora_A.device)
                        
                        if len(lora_index) > 0:
                            #print(f"cpjcpjcpj {lora_index}")
                            lora_A_copy = lora_A.clone()
                            lora_A[:, :, :] = lora_A[:, :, :] * 0
                            lora_A[:, :, rank*lora_index] = lora_A_copy[:, :, rank*lora_index]  
                            lora_A[:, :, rank*lora_index+rank-1] = lora_A_copy[:, :, rank*lora_index+rank-1]
                            lora_weight = self.lora_weight.weight.clone()
                            lora_weight = lora_weight * self.lora_weight_for_training
                            lora_weight_copy = self.lora_weight.weight.clone()
                            lora_weight_copy = lora_weight_copy * self.lora_weight_for_training
                            lora_weight[:, :] = -float('inf')
                            lora_weight[:, lora_index.to(lora_weight.device)] = lora_weight_copy[:, lora_index.to(lora_weight.device)]
                            if self.mask_answer_list is not None:
                                lora_weight[:, self.mask_answer_list] = -float('inf')
                            #lora_weight = F.softmax(5 * lora_weight.to(lora_A.device), dim=1) #base version
                            
                            #TODO:还是改成答案regularization好一点，阈值应该用来限制sequential问题中只有单个edit的情况（使用s key和r key）
                            #if not CAL_PROB:
                                #print(f"cpjcpjcpjzzz {lora_weight}")
                            
                            #TODO：这里是softmax方式获得lora_weight
                            #lora_weight_save = lora_weight.clone()
                            #lora_weight = F.softmax((100 * lora_weight).to(lora_A.device), dim=1)
                            #for t_i in range(lora_weight_save.shape[1]):
                                #if lora_weight_save[0][t_i] < 1.15:
                                    #lora_weight[0][t_i] = 0
                            #TODO：这里是top-1方式获得lora_weight
                            
                            #_, max_index = torch.topk(lora_weight[0], 2, largest=True, sorted=True)
                            #max_index = max_index[0:1]
                            
                            #大实验使用
                            
                            #max_value, max_index = torch.topk(torch.abs(lora_A_out_[0][0]), 10, largest=True, sorted=True)
                            max_value_1, max_index_1 = torch.topk(lora_weight[0], 4, largest=True, sorted=True)
                            '''
                            if not CAL_PROB:
                                print("cpjcpjcpjc[j]")
                                print([2*I, 2*I+1])
                                print(max_value)
                                print(max_index)
                            '''   
                            only_one_question_list = [2 * ONLY_ONE_QUESTION[only]-only for only in range(len(ONLY_ONE_QUESTION))]
                            if SEARCH:
                                max_index = list()
                                max_index_weight = list()
                                for t_i in range(lora_weight.shape[1]):
                                    if lora_weight[0][t_i] > 0.5 and t_i not in only_one_question_list and t_i in max_index_1.view(-1).detach().cpu().tolist():
                                        max_index.append(t_i)
                                        max_index_weight.append(lora_weight[0][t_i])
                                max_index = torch.tensor(max_index, device=lora_A.device)
                                MAX_INDEX = max_index
                                MAX_INDEX_WEIGHT = max_index_weight
                                print("cpjcpjcpjphm")
                                print(max_index)
                                print(max_index_weight)
                                #print(lora_weight[0])
                            else:
                                max_index = MAX_INDEX
                                max_index_weight = MAX_INDEX_WEIGHT
                                
                            
                            #小实验使用，大实验也使用
                            
                            if BEAM:
                                #max_index = torch.tensor([2*I-len(ONLY_ONE_QUESTION), 2*I+1-len(ONLY_ONE_QUESTION)], device=lora_A.device)
                                max_index = torch.tensor([I], device=lora_A.device)
                            else:
                                #max_index = torch.tensor([2*I-len(ONLY_ONE_QUESTION), 2*I+1-len(ONLY_ONE_QUESTION)], device=lora_A.device)
                                max_index = torch.tensor([I], device=lora_A.device)
                            

                            lora_weight[:, :] = 0
                            if len(max_index) > 0:
                                lora_weight[0][max_index] = 1
                            lora_weight = lora_weight.to(lora_A.device)
                            
                            
                            #if not CAL_PROB:
                                #print(f"cpjcpjcpj {lora_weight}")
                            
                            lora_A = lora_A.reshape(lora_A.shape[0], lora_A.shape[1], int(lora_A.shape[2] / rank), rank)
                            lora_A = torch.einsum('abcd,ec->abecd', lora_A, lora_weight)
                            lora_A = lora_A.reshape(lora_A.shape[0], lora_A.shape[1], -1)
                        else:
                            lora_A[:, :, :] = lora_A[:, :, :] * 0
                        
            '''
            if lora_A.shape[2] > 2 and not SAVE_REPRESENTATION:
                lora_A = torch.cat((lora_A, lora_A), dim=0)
                lora_B = torch.cat((lora_B, lora_B), dim=0)
            '''
            
            if True:
                if (lora_A.shape[2] > rank and not SAVE_REPRESENTATION and result.shape[1] == 1) or BEAM:
                    #print("cnmcnmcnmcnmcnmcnmcnmcnm")
                    #melo
                    #print("phmcpjphmcpj")
                    #print(lora_B.shape)
                    if not COMPARE:
                        #print("phmccccccc")
                        #print(LEN_VECTORDB)
                        
                        if LEN_VECTORDB % 2 == 0:
                            if lora_A.shape[2] > rank:
                                #if not CAL_PROB:
                                    #print("phmcpjcpjcpj")
                                    #print(max_index)
                                '''
                                lora_B[:, 2*torch.min(max_index):2*torch.min(max_index)+2, 0:int(lora_B.shape[2]*7/8)] = lora_B[:, 2*torch.min(max_index):2*torch.min(max_index)+2, 0:int(lora_B.shape[2]*7/8)] * 0
                                lora_B[:, 2*torch.max(max_index):2*torch.max(max_index)+2, int(lora_B.shape[2]*1/8):] = lora_B[:, 2*torch.max(max_index):2*torch.max(max_index)+2, int(lora_B.shape[2]*1/8):] * 0
                                '''
                                #print("plmcpjaibo")
                                #print(MASK_MATRIX.shape)
                                #print(MASK_MATRIX[torch.min(max_index)])
                                #print(MASK_MATRIX[torch.max(max_index)])
                                #print(lora_B[:, rank*torch.min(max_index):rank*torch.min(max_index)+rank, MASK_MATRIX_REAL[torch.min(max_index)]])
                                #print(lora_B[:, rank*torch.max(max_index):rank*torch.max(max_index)+rank, MASK_MATRIX_REAL[torch.min(max_index)]])
                                
                                lora_B[:, rank*torch.min(max_index):rank*torch.min(max_index)+rank, MASK_MATRIX[torch.min(max_index)]] = lora_B[:, rank*torch.min(max_index):rank*torch.min(max_index)+rank, MASK_MATRIX[torch.min(max_index)]] * 0
                                lora_B[:, rank*torch.max(max_index):rank*torch.max(max_index)+rank, MASK_MATRIX[torch.max(max_index)]] = lora_B[:, rank*torch.max(max_index):rank*torch.max(max_index)+rank, MASK_MATRIX[torch.max(max_index)]] * 0
                                
                                if not CAL_PROB:
                                    mask_1 = MASK_MATRIX_REAL[torch.min(max_index)].detach().cpu().numpy().tolist()
                                    mask_2 = MASK_MATRIX_REAL[torch.max(max_index)].detach().cpu().numpy().tolist()
                                    #only_1 = list()
                                    #only_2 = list()
                                    OVER = 0
                                    for mas in range(14336):
                                        if mas in mask_1 and mas in mask_2:
                                            OVER = OVER + 1
                                        #elif mas in mask_1:
                                        #    only_1.append(mas)
                                        #elif mas in mask_2:
                                        #    only_2.append(mas)
                                    print("test the together")
                                    print(OVER)
                                    #print(len(only_1))
                                    #print(len(only_2))
                                
                            else:
                                pass
                        else:
                            pass
                        
                    lora_A = nn.functional.normalize(lora_A, dim=1)
                    x = nn.functional.normalize(x, dim=2)
                    lora_A_out = self.lora_dropout[self.active_adapter](x) @ lora_A
                    '''
                    if (1000*lora_A_out[0][0][2])**2<100 and not COMPARE:
                        lora_A_out[0][0][2] = 0
                        lora_A_out[0][0][0] = -0.5494
                        lora_A_out[0][0][1] = 0.5691
                    if (1000*lora_A_out[0][0][3])**2<100 and not COMPARE:
                        lora_A_out[0][0][3] = 0
                    '''   
                    
                    
                    '''
                    if not COMPARE:
                        #print("phmccccccc")
                        #print(LEN_VECTORDB)
                        
                        if LEN_VECTORDB % 2 == 0:
                            if lora_A.shape[2] > rank:
                                lora_A_out = torch.zeros_like(lora_A_out, device=lora_A_out.device)
                                lora_A_out[:, :, rank*torch.min(max_index):rank*torch.min(max_index)+rank] = 1 
                                lora_A_out[:, :, rank*torch.max(max_index):rank*torch.max(max_index)+rank] = 1
                    '''
                    if not COMPARE:
                        result_1 = lora_A_out[:, :, rank*torch.min(max_index):rank*torch.min(max_index)+rank] @ lora_B[:, rank*torch.min(max_index):rank*torch.min(max_index)+rank, :]
                        result_2 = lora_A_out[:, :, rank*torch.max(max_index):rank*torch.max(max_index)+rank] @ lora_B[:, rank*torch.max(max_index):rank*torch.max(max_index)+rank, :]

                        result_final = result_1 + result_2
                        result_final[(result_1>torch.zeros_like(result_1)) * (result_2<torch.zeros_like(result_2))] = result_final[(result_1>torch.zeros_like(result_1)) * (result_2<torch.zeros_like(result_2))] * 0
                        result_final[(result_1<torch.zeros_like(result_1)) * (result_2>torch.zeros_like(result_2))] = result_final[(result_1<torch.zeros_like(result_1)) * (result_2>torch.zeros_like(result_2))] * 0

                        CONTR = 0
                        CONTR = CONTR + ((result_1>torch.zeros_like(result_1)) * (result_2<torch.zeros_like(result_2))).sum().item() + \
                        ((result_1<torch.zeros_like(result_1)) * (result_2>torch.zeros_like(result_2))).sum().item()

                        result = result + torch.einsum('km,bkd->bmd', mask, result_final \
                                * self.scaling[self.active_adapter])
                    else:
                        result = result + torch.einsum('km,bkd->bmd', mask, (lora_A_out @ lora_B) \
                                * self.scaling[self.active_adapter])
                    
                else:
                    #melo
                    
                    if SAVE_MASK:
                        
                        #print(result.shape) torch.Size([1, 8, 14336])
                        '''
                        for kkk in range(KEY_ID, result.shape[1]):
                            _, for_max_index = torch.topk(result[0][kkk]*result[0][kkk], 10, largest=True, sorted=True)
                            print(for_max_index)
                        '''

                        #取本来使用最多的
                        '''
                        result_for_mask = result[0][KEY_ID+1]*result[0][KEY_ID+1]
                        for kkk in range(KEY_ID+2, SUCCESS_ANSWERS_BEGIN):
                            result_for_mask = result_for_mask + result[0][kkk]*result[0][kkk]
                        _, for_max_index = torch.topk(result_for_mask, 12544, largest=False, sorted=True)
                        _, for_max_index_real = torch.topk(result_for_mask, 1792, largest=True, sorted=True)
                        _, for_max_index_real_includeinput = torch.topk(result[0][KEY_ID]*result[0][KEY_ID], 1792, largest=True, sorted=True)
                        '''
                        #尝试取对其他的推动最少的
                        
                        #print(DOWN_PROJ_W.T.shape)
                        #print(W.shape)
                        input_1 = NORM(DOWN_PROJ_W.T)
                        
                        for_max_index_list = list(range(14336))
                        for_max_index_real_list = list()
                        for_max_index_real_includeinput_list = list()

                        #size_1 = 14336 // LABEL.shape[0]
                        #the = 14336 % LABEL.shape[0]


                        for kkk in range(LABEL.shape[0]):
                            result_for_mask = torch.einsum('km,m->k', input_1, W[LABEL[kkk]])
                            _, for_max_index_real_temp = torch.topk(result_for_mask, 896*1, largest=False, sorted=True)
                            for_max_index_real_temp = for_max_index_real_temp[896*0:896*1]
                            #for_max_index_real_temp = torch.tensor(random.sample(list(range(14336)), 112), device=result_for_mask.device)
                            _, for_max_index_real_includeinput_temp = torch.topk(result_for_mask, 896*1, largest=False, sorted=True)
                            for_max_index_real_includeinput_temp = for_max_index_real_includeinput_temp[896*0:896*1]
                            #for_max_index_real_includeinput_temp = for_max_index_real_temp
                            for_max_index_real_list = for_max_index_real_list + for_max_index_real_temp.detach().cpu().numpy().tolist()
                            for_max_index_real_includeinput_list = for_max_index_real_includeinput_list + for_max_index_real_includeinput_temp.detach().cpu().numpy().tolist()
                        
                        for_max_index_list = set(for_max_index_list) - set(for_max_index_real_list)
                        for_max_index_real_list = set(for_max_index_real_list)
                        for_max_index_real_includeinput_list = set(for_max_index_real_includeinput_list)

                        for_max_index = torch.tensor(list(for_max_index_list), device=result.device)
                        for_max_index_real = torch.tensor(list(for_max_index_real_list), device=result.device)
                        for_max_index_real_includeinput = torch.tensor(list(for_max_index_real_includeinput_list), device=result.device)
                        
                        
                        #尝试取对其他推动最小的，并保证取到的数量相同
                        '''
                        input_1 = NORM(DOWN_PROJ_W.T)
                        
                        for_max_index_list = list(range(14336))
                        for_max_index_real_list = list()
                        for_max_index_real_includeinput_list = list()

                        ex = 0
                        
                        while True:
                            for kkk in range(LABEL.shape[0]):
                                result_for_mask = torch.einsum('km,m->k', input_1, W[LABEL[kkk]])
                                _, for_max_index_real_temp = torch.topk(result_for_mask, ex+1, largest=False, sorted=True)
                                for_max_index_real_temp = for_max_index_real_temp[ex:ex+1]
                                #for_max_index_real_temp = torch.tensor(random.sample(list(range(14336)), 112), device=result_for_mask.device)
                                _, for_max_index_real_includeinput_temp = torch.topk(result_for_mask, ex+1, largest=False, sorted=True)
                                for_max_index_real_includeinput_temp = for_max_index_real_includeinput_temp[ex:ex+1]
                                #for_max_index_real_includeinput_temp = for_max_index_real_temp
                                for_max_index_real_list = for_max_index_real_list + for_max_index_real_temp.detach().cpu().numpy().tolist()
                                for_max_index_real_includeinput_list = for_max_index_real_includeinput_list + for_max_index_real_includeinput_temp.detach().cpu().numpy().tolist()
                            
                            ex = ex + 1
                            for_max_index_real_list = list(set(for_max_index_real_list))
                            for_max_index_real_includeinput_list = list(set(for_max_index_real_includeinput_list))
                            if len(for_max_index_real_list) >= 896:
                                break
                        
                        for_max_index_real_list = for_max_index_real_list[0:896]
                        for_max_index_real_includeinput_list = for_max_index_real_includeinput_list[0:896]
                        for_max_index_list = set(for_max_index_list) - set(for_max_index_real_list)
                        for_max_index_real_list = set(for_max_index_real_list)
                        for_max_index_real_includeinput_list = set(for_max_index_real_includeinput_list)

                        for_max_index = torch.tensor(list(for_max_index_list), device=result.device)
                        for_max_index_real = torch.tensor(list(for_max_index_real_list), device=result.device)
                        for_max_index_real_includeinput = torch.tensor(list(for_max_index_real_includeinput_list), device=result.device)
                        '''
                        
                        #取头尾
                        '''
                        if LEN_VECTORDB % 2 == 0:
                            for_max_index = torch.tensor(list(range(int(lora_B.shape[2]*1/4), lora_B.shape[2])), device=for_max_index.device)
                        else:
                            for_max_index = torch.tensor(list(range(0, int(lora_B.shape[2]*3/4))), device=for_max_index.device)
                        '''
                        
                        if MASK_MATRIX is None:
                            '''
                            MASK_MATRIX = for_max_index.unsqueeze(0)
                            MASK_MATRIX_REAL = for_max_index_real.unsqueeze(0)
                            MASK_MATRIX_REAL_INCLUDEINPUT = for_max_index_real_includeinput.unsqueeze(0)
                            '''
                            MASK_MATRIX = [for_max_index]
                            MASK_MATRIX_REAL = [for_max_index_real]
                            MASK_MATRIX_REAL_INCLUDEINPUT = [for_max_index_real_includeinput]
                            
                        else:
                            '''
                            MASK_MATRIX = torch.cat((MASK_MATRIX, for_max_index.unsqueeze(0)), dim=0)
                            MASK_MATRIX_REAL = torch.cat((MASK_MATRIX_REAL, for_max_index_real.unsqueeze(0)), dim=0)
                            MASK_MATRIX_REAL_INCLUDEINPUT = torch.cat((MASK_MATRIX_REAL_INCLUDEINPUT, for_max_index_real_includeinput.unsqueeze(0)), dim=0)
                            '''
                            MASK_MATRIX.append(for_max_index)
                            MASK_MATRIX_REAL.append(for_max_index_real)
                            MASK_MATRIX_REAL_INCLUDEINPUT.append(for_max_index_real_includeinput)
                            

                    '''
                    if LEN_VECTORDB % 2 == 0:
                        if lora_A.shape[2] > 2:
                            lora_B[:, -2:, int(lora_B.shape[2]*1/8):] = lora_B[:, -2:, int(lora_B.shape[2]*1/8):] * 0
                        else:
                            lora_B[:, :, int(lora_B.shape[2]*1/8):] = lora_B[:, :, int(lora_B.shape[2]*1/8):] * 0
                    else:
                        if lora_A.shape[2] > 2:
                            lora_B[:, -2:, 0:int(lora_B.shape[2]*7/8)] = lora_B[:, -2:, 0:int(lora_B.shape[2]*7/8)] * 0
                        else:
                            lora_B[:, :, 0:int(lora_B.shape[2]*7/8)] = lora_B[:, :, 0:int(lora_B.shape[2]*7/8)] * 0
                    '''
                    if lora_A.shape[2] > rank:
                        lora_B[:, -rank:, MASK_MATRIX[-1]] = lora_B[:, -rank:, MASK_MATRIX[-1]] * 0
                    else:
                        lora_B = lora_B.repeat(x.shape[0],1,1)
                        for ba in range(lora_B.shape[0]):
                            lora_B[ba, :, MASK_MATRIX[-1]] = lora_B[ba, :, MASK_MATRIX[-1]] * 0
                    
                    lora_A = nn.functional.normalize(lora_A, dim=1)
                    x = nn.functional.normalize(x, dim=2)
                    lora_A_out = self.lora_dropout[self.active_adapter](x) @ lora_A

                    if True:
                        try:
                            #print(lora_A.shape)
                            #print(torch.norm(lora_A, p=2, dim=2).shape)
                            #print(lora_A_out.shape)
                            #TODO: 让答案中间的token很大，两边很小，后续很小
                            #TODO: 稀疏(finish_loss到底如何起作用)
                            #TODO: double edit还有没有用
                            #TODO: 因果推断稀疏
                            #TODO: 压缩
                            FINISH_LOSS_POS = 0
                            FINISH_LOSS_NEG = 0
                            FINISH_LOSS_HIDDEN = 0
                            '''
                            for fin in range(SUCCESS_ANSWERS_BEGIN-1, lora_A_out.shape[1]):
                                FINISH_LOSS += torch.mean(torch.norm(lora_A_out[:, fin, :], p=2, dim=1))# - torch.mean(lora_A_out[:, KEY_ID, :]) / torch.norm(lora_A, p=2, dim=1)[0][0] - torch.mean(lora_A_out[:, KEY_ID, :]) / torch.norm(lora_A, p=2, dim=1)[0][1]
                            '''
                            if FINISH == 1:
                                
                                for sam in range(lora_A_out.shape[0]):
                                    if sam == 0:
                                        for fin in range(SUCCESS_ANSWERS_BEGIN-1, SUCCESS_ANSWERS_BEGIN):
                                        #for fin in range(SUCCESS_ANSWERS_BEGIN-1, lora_A_out.shape[1]):
                                            FINISH_LOSS_POS += torch.mean(torch.norm(lora_A_out[sam:sam+1, fin, :], p=2, dim=1)) / (lora_A_out.shape[1]-SUCCESS_ANSWERS_BEGIN+1)
                                    
                                        for fin in range(KEY_ID, SUCCESS_ANSWERS_BEGIN-1):
                                            FINISH_LOSS_NEG -= torch.mean(torch.norm(lora_A_out[sam:sam+1, fin, :], p=2, dim=1)) / (lora_A_out.shape[1]-SUCCESS_ANSWERS_BEGIN+1)
                            elif FINISH == 2:

                                for sam in range(lora_A_out.shape[0]):
                                    for fin in range(KEY_ID+1, lora_A_out.shape[1]):
                                        #print(KEY_ID)
                                        #print(NUM_SUCCESS_ANSWERS)
                                        FINISH_LOSS_POS += torch.mean(torch.norm(lora_A_out[sam:sam+1, fin, :], p=2, dim=1)) / (lora_A_out.shape[1]-KEY_ID.to(lora_A_out.device)-1) / (NUM_SUCCESS_ANSWERS)
                                '''
                                elif sam > 0 and sam <= NUM_SUCCESS_ANSWERS:
                                    for fin in range(KEY_ID+1, lora_A_out.shape[1]):
                                        #print(KEY_ID)
                                        #print(NUM_SUCCESS_ANSWERS)
                                        FINISH_LOSS_POS += torch.mean(torch.norm(lora_A_out[sam:sam+1, fin, :], p=2, dim=1)) / (lora_A_out.shape[1]-KEY_ID.to(lora_A_out.device)-1) / (NUM_SUCCESS_ANSWERS)
                                '''
                                '''
                                elif sam > 0 and sam < lora_A_out.shape[0]-1-NUM_SUCCESS_ANSWERS:
                                    for fin in range(SUCCESS_ANSWERS_BEGIN, lora_A_out.shape[1]):
                                        FINISH_LOSS_NEG -= torch.mean(torch.norm(lora_A_out[sam:sam+1, fin, :], p=2, dim=1)) / (lora_A_out.shape[1]-SUCCESS_ANSWERS_BEGIN) / (lora_A_out.shape[0]-2-NUM_SUCCESS_ANSWERS)
                                '''
                            
                            #print("jjjjjjjjjjjjjjjj")
                            #print(KEY_ID)
                            #print(torch.norm(lora_A, p=2, dim=1).shape)
                            #print(torch.mean(torch.norm(lora_A_out[:, -1, :], p=2, dim=1)))
                            #print( - torch.mean(lora_A_out[:, KEY_ID, :]) / torch.norm(lora_A, p=2, dim=1)[0][0] - torch.mean(lora_A_out[:, KEY_ID, :]) / torch.norm(lora_A, p=2, dim=1)[0][1])
                            
                        except:
                            pass
                    result = result + torch.einsum('km,bkd->bmd', mask, (lora_A_out @ lora_B) \
                            * self.scaling[self.active_adapter])
                    '''
                    if "up_proj" in self.key:
                        #print(self.key)
                        extra_weight = self.lora_dropout[self.active_adapter](x) @ lora_A
                    elif "down_proj" in self.key:
                        #print(self.key)
                        result = result + torch.einsum('km,bkd->bmd', mask, (extra_weight @ lora_B) \
                            * self.scaling[self.active_adapter])
                        extra_weight = None
                    '''

            if SAVE_REPRESENTATION:
                pass
                '''
                if self.key == "model.layers.31.mlp.down_proj":
                    if SAVE_REPRESENTATION_ANSWER:
                        ANSWER_KEY_ID = [int((SUCCESS_ANSWERS_BEGIN-1-KEY_ID)/2)+KEY_ID-1]#[result.shape[1]-2]#list(range(KEY_ID, result.shape[1]-1))
                        ANSWER_KEY_ID_REAL = list(range(KEY_ID, result.shape[1]-1))
                        representation_temp = result[0:1, ANSWER_KEY_ID[0]]
                        ANSWER_KEY_ID_LEN_LIST.append(len(ANSWER_KEY_ID))
                        ANSWER_KEY_ID_LEN_LIST_REAL.append(len(ANSWER_KEY_ID_REAL))
                        for a_k in range(len(ANSWER_KEY_ID)):
                            if a_k > 0:
                                representation_temp = representation_temp + result[0:1, ANSWER_KEY_ID[a_k]]
                        representation_temp = representation_temp / len(ANSWER_KEY_ID)
                        if REPRESENTATION_MATRIX is None:
                            REPRESENTATION_MATRIX = representation_temp
                        else:
                            REPRESENTATION_MATRIX = torch.cat((REPRESENTATION_MATRIX, representation_temp), dim=0)
                    if SAVE_REPRESENTATION_SR:
                        if REPRESENTATION_SUBJECT_MATRIX is None:
                            REPRESENTATION_SUBJECT_MATRIX = result[0:1, SUBJECT_KEY_ID]
                        else:
                            REPRESENTATION_SUBJECT_MATRIX = torch.cat((REPRESENTATION_SUBJECT_MATRIX, result[0:1, SUBJECT_KEY_ID]), dim=0)
                        if REPRESENTATION_RELATION_MATRIX is None:
                            REPRESENTATION_RELATION_MATRIX = result[0:1, KEY_ID]
                        else:
                            REPRESENTATION_RELATION_MATRIX = torch.cat((REPRESENTATION_RELATION_MATRIX, result[0:1, KEY_ID]), dim=0)
                '''        
            

            if lora_A.shape[2] > rank and not SAVE_REPRESENTATION:
                pass
                '''
                if self.key == "model.layers.31.mlp.down_proj":
                    for index in range(result.shape[1]):
                        cos_knowledge = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_MATRIX, dim=1))
                        if not DECODE_TRAINING:
                            if COS_KNOWLEDGE_MATRIX is None:
                                COS_KNOWLEDGE_MATRIX = cos_knowledge.detach().clone()
                            else:
                                COS_KNOWLEDGE_MATRIX = torch.cat((COS_KNOWLEDGE_MATRIX, cos_knowledge.detach().clone()), dim=0)
                        cos_knowledge_subject = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_SUBJECT_MATRIX, dim=1))
                        if not DECODE_TRAINING:
                            if COS_KNOWLEDGE_SUBJECT_MATRIX is None:
                                COS_KNOWLEDGE_SUBJECT_MATRIX = cos_knowledge_subject.detach().clone()
                            else:
                                COS_KNOWLEDGE_SUBJECT_MATRIX = torch.cat((COS_KNOWLEDGE_SUBJECT_MATRIX, cos_knowledge_subject.detach().clone()), dim=0) 
                        #print(COS_KNOWLEDGE_SUBJECT_MATRIX)   
                        #print(COS_KNOWLEDGE_SUBJECT_MATRIX.shape)                        
                        cos_knowledge_relation = torch.einsum('bd,cd->bc', nn.functional.normalize(result[0:1, index], dim=1), nn.functional.normalize(REPRESENTATION_RELATION_MATRIX, dim=1))
                        if not DECODE_TRAINING:
                            if COS_KNOWLEDGE_RELATION_MATRIX is None:
                                COS_KNOWLEDGE_RELATION_MATRIX = cos_knowledge_relation.detach().clone()
                            else:
                                COS_KNOWLEDGE_RELATION_MATRIX = torch.cat((COS_KNOWLEDGE_RELATION_MATRIX, cos_knowledge_relation.detach().clone()), dim=0)
                '''
            '''
            if result.shape[1] == 1:
                if self.key == "model.layers.31.mlp.down_proj":
                    DECODE_LOSS = torch.mean(cos_knowledge)
            '''
            
            ratio = None
            ratio_sum = None
            if result.shape[1] == 1 and not COMPARE:
                pass
                '''
                if self.key == "model.layers.31.mlp.down_proj":
                    #print(COS_KNOWLEDGE_MATRIX)
                    for i in range(len(ANSWER_KEY_ID_LEN_LIST)):
                        k_values, _ = torch.topk(COS_KNOWLEDGE_MATRIX[:, i], min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX[:, i].shape[0]), largest=True, sorted=True)
                        k_values = k_values.mean().unsqueeze(0)
                        s_values = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX[:, i], dim=0, keepdim=True)[0]
                        s_values = s_values.mean().unsqueeze(0) 
                        r_values = torch.max(COS_KNOWLEDGE_RELATION_MATRIX[:, i], dim=0, keepdim=True)[0]
                        r_values = r_values.mean().unsqueeze(0)
                        k_values_pre, _ = torch.topk(COS_KNOWLEDGE_MATRIX[:, i], min(ANSWER_KEY_ID_LEN_LIST[i]-1, COS_KNOWLEDGE_MATRIX[:, i].shape[0]), largest=True, sorted=True)
                        #print(f"cnmcnmcnm{k_values_pre}")
                        k_values_pre = (k_values_pre.sum().unsqueeze(0) + cos_knowledge[:, i]) / min(ANSWER_KEY_ID_LEN_LIST[i], COS_KNOWLEDGE_MATRIX[:, i].shape[0])
                        s_values_pre = torch.max(COS_KNOWLEDGE_SUBJECT_MATRIX[:, i], dim=0, keepdim=True)[0]
                        s_values_pre = s_values_pre.mean().unsqueeze(0) 
                        r_values_pre = torch.max(COS_KNOWLEDGE_RELATION_MATRIX[:, i], dim=0, keepdim=True)[0]
                        r_values_pre = r_values_pre.mean().unsqueeze(0)
                        
                        if ratio is None:
                            ratio = -(k_values**1) * (s_values * r_values)**1
                            ratio_sum = -k_values_pre * (s_values_pre * r_values_pre)**5
                        else:
                            ratio = torch.cat((ratio, -(k_values**1) * (s_values * r_values)**1), dim=0)
                            #print(f"cpjhhhhhcpjhhhhh{ratio.shape}")
                            ratio_sum = torch.cat((ratio_sum, -k_values_pre * (s_values_pre * r_values_pre)**5), dim=0)
                            #print(f"cpjhhhhcpjhhhh{k_values_sum.shape}")

                    v = -F.relu(-ratio_sum) #loss version 1
                    #v = -F.relu(-ratio_sum * self.lora_weight_for_training) #loss version 2
                    print(f"cpjhhhcpjhh[ph{v}")
                    #print((k_values_sum * cos_knowledge_subject * cos_knowledge_relation + ratio).shape)
                    if torch.max((-ratio_sum + ratio), dim=0, keepdim=True)[0] > 0:
                        DECODE_LOSS = torch.mean(-F.relu(-ratio_sum)) #loss version 1
                        #DECODE_LOSS = torch.mean(-F.relu(-ratio_sum * self.lora_weight_for_training)) #loss version 2
                    else:
                        DECODE_LOSS = torch.mean(-F.relu(-ratio_sum)) #loss version 1
                        #DECODE_LOSS = torch.mean(-F.relu(-ratio_sum * self.lora_weight_for_training)) #loss version 2
                    # / (-ratio / torch.max((-ratio), dim=0, keepdim=True)[0])
                '''
            #print(result[0][-1])
            #print("cnmcnmcnmcnmcnm")
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        #if self.key == "model.layers.31.down_proj":
        #print(x.shape)
        #print(result.shape)

        return result


class PatchLinear(nn.Linear, PatchLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        type: str = "up_proj",
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_rank_per_block: int = 1,
        memory=None,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        PatchLayer.__init__(self, in_features=in_features, out_features=out_features, type=type)
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, num_rank_per_block)
        self.lora_block_mapping = LORA_BLOCK_MAPPING
        self.active_adapter = adapter_name
        self.memory = memory

    def forward(self, x: torch.Tensor):
        global MEMORYLOSS       
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        previous_dtype = x.dtype
        if self.active_adapter not in self.patch.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        if self.r[self.active_adapter] > 0:
            #result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.type == "up_proj" or self.type == "gate_proj":
                
                result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
                patch_weight, patch_bias = self.nd_patch(self.patch[self.active_adapter].weight, self.patch[self.active_adapter].bias)
                add_result = F.linear(x, patch_weight, bias=None)
                memory_result = F.linear(self.memory, patch_weight, bias=None)
                add_result_reshape = add_result[0][0].unsqueeze(0).repeat(memory_result.shape[0], 1)
                act_loss = 0.0 - add_result_reshape
                act_loss = torch.exp(act_loss)
                #act_loss = torch.topk(act_loss, k=100)[0]
                act_loss = torch.mean(act_loss)
                #print(memory_result.shape)
                #print(add_result.shape)
                l1_loss = memory_result - add_result_reshape - 0.0
                l1_loss = torch.exp(l1_loss)
                #print(l1_loss.shape)
                l1_loss = torch.mean(l1_loss, dim=1)
                #print(l1_loss.shape)
                l1_loss = torch.topk(l1_loss, k=100)[0]
                l1_loss = torch.mean(l1_loss)
                l2_loss = memory_result - 0.0
                l2_loss = torch.exp(l2_loss)
                #print(l2_loss.shape)
                l2_loss = torch.mean(l2_loss, dim=1)
                #print(l2_loss.shape)
                l2_loss = torch.topk(l2_loss, k=100)[0]
                l2_loss = torch.mean(l2_loss)
                MEMORYLOSS += act_loss + l1_loss + l2_loss
                #result += add_result
                result = torch.cat((result, add_result), dim=2)
                #print(result.shape)
            elif self.type == "down_proj":
                patch_weight, patch_bias = self.nd_patch(self.patch[self.active_adapter].weight.T, self.patch[self.active_adapter].bias.T)
                #print(self.weight.shape)
                #print(patch_weight.shape)
                weight_all = torch.cat((self.weight, patch_weight.T), dim=1)
                #print(weight_all.shape)
                #print(self.bias.shape)
                #print(patch_bias.T.shape)
                #bias_all = torch.cat((self.bias, patch_bias.T), dim=1)
                #print(bias_all.shape)
                result = F.linear(x, transpose(weight_all, self.fan_in_fan_out), bias=None)
                #result += F.linear(x, patch_weight, bias=patch_bias)
            #print(x.shape)
            #print((self.lora_dropout[self.active_adapter](x) @ lora_A).shape)
            #result += (self.lora_dropout[self.active_adapter](x) @ lora_A @ lora_B) \
                      #* self.scaling[self.active_adapter]
            #print("cnmcnmcnmcnmcnm")
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result

class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter].weight
                        @ self.lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data += (
                    self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                    F.conv2d(
                        self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                        self.lora_B[self.active_adapter].weight,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[self.active_adapter]
                )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data -= (
                    self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                    F.conv2d(
                        self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                        self.lora_B[self.active_adapter].weight,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[self.active_adapter]
                )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                        )
                    result += output
                return result




