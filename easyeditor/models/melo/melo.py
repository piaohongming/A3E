from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os
from ..losses import masked_log_probs

from torch.nn import Parameter
from easyeditor.util import nethook

from .util import *


from .peft_egg import (
    PeftModel,
    prepare_model_for_int8_training,
    MeloConfig,
    TpatcherConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from .peft_egg.src.peft.tuners.melo import LoraLayer, GraceLayer, PatchLayer
# from hooks import lora_backward_hook
from .models import BertClassifier

METHOD = "grace" #t-patcher, grace, melo_origin
LOG = logging.getLogger(__name__)
def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)

class LORA(torch.nn.Module):
    def __init__(self, model, config, model_tok,memory=None, scale=None):
        super(LORA, self).__init__()
        self.config = config

        

        '''Apply_lora
        '''            
        r_num = config.grace.num_block * config.grace.num_rank_per_block
        self.lora_config = MeloConfig(
            r = r_num,
            lora_alpha = r_num,
            target_modules= list(config.model.target_modules),
            lora_dropout = config.lora.lora_dropout,
            task_type = config.lora_task_type,
            fan_in_fan_out= config.model.fan_in_fan_out,
            grace_layer = config.model.grace_layer,
            grace_config= config.grace.to_dict()
        )
        self.log_dict = {}

        '''Load
        '''
        # self.original_model = model
        # self.model = model

        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config, memory=memory)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_lora_modules()
        self.grace_layer = self.named_grace_layer()
        # self.register_lora_backward_hooks(lora_backward_hook)

        '''Load Tokenizer
        '''
        self.model_tok = model_tok
        #self.classifier_tok = transformers.AutoTokenizer.from_pretrained(config.lora.cls_name)

        '''Parameters to be optimized
        '''
        self.opt_params = self.optim_parameters()
        self.forward = self.model.model.forward

        self.KLDiv = torch.nn.KLDivLoss(reduction="none")

        def _edit_loss_fn(config, pred, targ, **kwargs):
            if 'minigpt4' in config.model_name.lower() or 'blip' in self.config.model_name.lower():
                return masked_log_probs(config, pred, targ, exact_match=self.config.exact_match, shift=True, **kwargs)
            elif 't5' in config.model_class.lower():
                return masked_log_probs(config, pred, targ,)
            elif 'gpt' in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True, **kwargs)
            elif 'llama' in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True, **kwargs)
            elif 'internlm' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'chatglm' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'qwen' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'mistral' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            else:
                return masked_log_probs(config, pred, targ,)

        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = masked_log_probs

        pass


    def optim_parameters(self):
        for name, param in self.model.named_parameters():
            #print(name)
            if param.requires_grad==True and 'lora' not in name:
                param.requires_grad = False
            elif 'lora' in name:
                param.requires_grad = True
                #print("lora: {}".format(name))
        lora_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        return lora_params

    def decode_optim_parameters(self):
        for name, param in self.model.named_parameters():
            #print(name)
            if param.requires_grad==True and ('lora_weight' not in name):
                param.requires_grad = False
            elif 'lora_weight' in name:
                #print(f"nmbnmbnmbnmb{name}")
                param.requires_grad = True
                #print("lora: {}".format(name))
        lora_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        return lora_params
    
    def print_decode_optim_parameters(self):
        for name, param in self.model.named_parameters():
            #print(name)
            if 'lora_weight' not in name:
                pass
            else:
                print(param)
                #print("lora: {}".format(name))


    #TODO
    def load_from_checkpoint(self, save_path):
        print(save_path)


    def save_classifier_weights(self,cls_dir):
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir,exist_ok=True)
        torch.save(self.classifier.state_dict(),f"{cls_dir}/classifier.pt")
    def save_lora_weights(self,lora_dir):
        self.model.save_pretrained(lora_dir+"/lora_checkpoint")


    def reset_lora(self):
        for key in self.lora_list:
            self.model.get_submodule(key).reset_lora_parameters('default')

    def named_lora_modules(self):
        module_list = [key for key,_ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key),LoraLayer):
                lora_list.append(key)
        return lora_list

    def named_grace_layer(self) -> str:
        module_list = [key for key, _ in self.model.named_modules()]
        grace_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key), GraceLayer):
                grace_list.append(key)
        return grace_list
        #assert len(grace_list) == 1, "At Most One Grace Layer"
        #return grace_list[0]

    def register_lora_backward_hooks(self,backward_hook_fn):
        for key in self.lora_list:
            self.model.get_submodule(key).register_backward_hook(backward_hook_fn)


    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()
        self.model.base_model.disable_grace_layer()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()
        self.model.base_model.enable_grace_layer()

    def _nor_loss(self, logits, dg_logits, tau=3):
        pred_probs = F.log_softmax(logits / tau, dim=1)
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / tau, dim=1)
        loss = (tau ** 2) * self.KLDiv(pred_probs, dg_probs)
        loss = torch.mean(loss)

        return loss
    
    
    def _ntd_loss(self, logits, dg_logits, label, tau=3):
        logits = torch.cat((logits[:, 0:label],logits[:, label+1:]), dim=1)
        dg_logits = torch.cat((dg_logits[:, 0:label],dg_logits[:, label+1:]), dim=1)
        pred_probs = F.log_softmax(logits / tau, dim=1)
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / tau, dim=1)
        loss = (tau ** 2) * self.KLDiv(pred_probs, dg_probs)
        loss = torch.mean(loss)
        return loss
    
    def _fair_loss(self, logits, label):
        loss = 0
        for i in range(label.shape[1]):
            logits_temp = torch.cat((logits[:, i:i+1, 0:label[0][i]], logits[:, i:i+1, label[0][i]+1:]), dim=2)
            loss += torch.mean(torch.sum(F.normalize(logits_temp, dim=2), dim=2)) / label.shape[1]
        return loss
    
    def _fair_loss_distribution(self, logits, label, tau=0.05):
        loss = 0
        dg_logits = torch.ones_like(logits, device=logits.device)
        for i in range(label.shape[1]):
            logits_temp = torch.cat((logits[:, i:i+1, 0:label[0][i]], logits[:, i:i+1, label[0][i]+1:]), dim=2)       
            dg_logits_temp = torch.cat((dg_logits[:, i:i+1, 0:label[0][i]], dg_logits[:, i:i+1, label[0][i]+1:]), dim=2)
            probs_temp = F.log_softmax(logits_temp / tau, dim=2)
            with torch.no_grad():
                dg_probs_temp = torch.softmax(dg_logits_temp / tau, dim=2)
            loss += torch.mean((tau ** 2) * self.KLDiv(probs_temp, dg_probs_temp)) / label.shape[1]

        return loss
    '''
    def _fair_cross_entropy():
    '''

    def _low_loss(self, logits, label):
        loss = 0
        logits = F.relu(logits)
        for i in range(label.shape[1]):
            logits_temp = torch.cat((logits[:, i:i+1, 0:label[0][i]], logits[:, i:i+1, label[0][i]+1:]), dim=2)
            loss += torch.mean(torch.sum(logits_temp, dim=2)) / label.shape[1]
        return loss
    
    def _fairmax_loss(self, logits, label):
        loss = 0
        for i in range(label.shape[1]):
            logits_temp = (logits[:, i:i+1, label[0][i]]-18)*(logits[:, i:i+1, label[0][i]]-18)
            loss += torch.mean(logits_temp) / label.shape[1]
        return loss
    
    def _judge_loss(self, logits, label):
        #print(logits)
        
        loss = 0
        for i in range(logits.shape[1]):
            logits_temp = logits[:, i:i+1, :]
            loss += torch.mean(logits_temp) / logits.shape[1]
        return loss

    def edit(self, tokens, tokens_withsuccessanswer, subject_tok, relation_tok, answer_tok, success_answers_begin, num_success_answers):
        def patch_save_representation(x, layer):
            if layer in ["model.layers.5.mlp.down_proj"]:
                #print("cpjcnmcnmcnmcnmcnmcnmcnm")
                
                self.model.base_model.save_representation_value(x)
            return x

        if "instruct" in self.config.model_name.lower():
            key_id = (tokens["labels"][0] == -100).sum()
        else:
            key_id = (tokens["labels"][0] == -100).sum() - 1
        print("cnmcnmcnmcnm:{}".format(key_id))
        #print(tokens)
        #print(subject_tok)
        self.model.base_model.set_key_id(key_id, subject_tok, relation_tok, answer_tok)
        self.model.base_model.set_num_success_answers(num_success_answers)

        #for key in self.model.base_model.state_dict().keys():
            #print(key)
        W = copy.deepcopy(self.model.base_model.model.state_dict()['lm_head.weight'])
        Norm_W = copy.deepcopy(self.model.base_model.model.state_dict()['model.norm.weight'])
        Norm = copy.deepcopy(self.model.base_model.model.model.norm)
        Down_proj_W = copy.deepcopy(self.model.base_model.model.state_dict()['model.layers.31.mlp.down_proj.weight'])

        self.model.base_model.set_W_Norm(W, Norm, Norm_W, Down_proj_W, tokens["labels"][0][key_id+1:success_answers_begin])
        

        optimizer = torch.optim.Adam(self.optim_parameters(), self.config.grace.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        for i in range(len(self.grace_layer)):
            setattr(self.model.get_submodule(self.grace_layer[i]), "training", True)
            setattr(self.model.get_submodule(self.grace_layer[i]), "edit_label", tokens_withsuccessanswer["labels"])
            setattr(self.model.get_submodule(self.grace_layer[i]), "key_id", key_id)


        self.losses = []
        
        #TODO: mask构造
        '''
        with torch.no_grad():
            self.model.base_model.save_mask(True)
            _ = self.model.model(**tokens_withsuccessanswer)
            self.model.base_model.save_mask(False)
        '''
        

        origin_distribution = None
        for i in range(50):
        #for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            for j in range(len(self.grace_layer)):
                setattr(self.model.get_submodule(self.grace_layer[j]), "batch_iter", i)
            if i == 0:
                self.model.base_model.save_mask(True)
            else:
                self.model.base_model.save_mask(False)

            # --- pass tokens through model (including through the GRACE layer) ---
            #print("nmknmknmknmknmk")
            #print(tokens_withsuccessanswer["input_ids"][0])
            self.model.base_model.save_representation(False, success_answers_begin)
            tokens_withsuccessanswer_clone = copy.deepcopy(tokens_withsuccessanswer)
            with torch.no_grad():
                for cur_pos in range(1):
                    outputs = self.model.model(**tokens_withsuccessanswer_clone)
                    #避免保存过多的mask_matrix和vector database
                    self.model.base_model.save_mask(False)
                    for j in range(len(self.grace_layer)):
                        setattr(self.model.get_submodule(self.grace_layer[j]), "batch_iter", i+1)
                    
                    next_token = torch.argmax(outputs.logits[:, -1], dim=-1)
                    #next_token[:] = 11
                    tokens_withsuccessanswer_clone["input_ids"] = torch.cat((tokens_withsuccessanswer_clone["input_ids"], next_token.unsqueeze(1)), dim=1)
                    tokens_withsuccessanswer_clone["labels"] = torch.cat((tokens_withsuccessanswer_clone["labels"], next_token.unsqueeze(1)), dim=1)
                    tokens_withsuccessanswer_clone["labels"][:, -1] = -100
                    tokens_withsuccessanswer_clone["attention_mask"] = torch.cat((tokens_withsuccessanswer_clone["attention_mask"], torch.ones((tokens_withsuccessanswer_clone["attention_mask"].shape[0], 1), device=tokens_withsuccessanswer_clone["attention_mask"].device)), dim=1)
            tokens_withsuccessanswer_clone["input_ids"] = torch.cat((tokens_withsuccessanswer_clone["input_ids"], tokens_withsuccessanswer_clone["input_ids"][0:1]), dim=0)
            tokens_withsuccessanswer_clone["labels"] = torch.cat((tokens_withsuccessanswer_clone["labels"], tokens_withsuccessanswer_clone["labels"][0:1]), dim=0)
            tokens_withsuccessanswer_clone["attention_mask"] = torch.cat((tokens_withsuccessanswer_clone["attention_mask"], tokens_withsuccessanswer_clone["attention_mask"][0:1]), dim=0)
            tokens_withsuccessanswer_clone["labels"][-1, :] = -100
            for ma in range(key_id, success_answers_begin):
                tokens_withsuccessanswer_clone["attention_mask"][-1][ma] = 0
            
            tokens_withsuccessanswer_clone_1 = copy.deepcopy(tokens_withsuccessanswer_clone)
            tokens_withsuccessanswer_clone_2 = copy.deepcopy(tokens_withsuccessanswer_clone)
            tokens_withsuccessanswer_clone_1["input_ids"] = torch.cat((tokens_withsuccessanswer_clone["input_ids"][0:1], tokens_withsuccessanswer_clone["input_ids"][num_success_answers+1:]), dim=0)
            tokens_withsuccessanswer_clone_1["labels"] = torch.cat((tokens_withsuccessanswer_clone["labels"][0:1], tokens_withsuccessanswer_clone["labels"][num_success_answers+1:]), dim=0)
            tokens_withsuccessanswer_clone_1["attention_mask"] = torch.cat((tokens_withsuccessanswer_clone["attention_mask"][0:1], tokens_withsuccessanswer_clone["attention_mask"][num_success_answers+1:]), dim=0)
            tokens_withsuccessanswer_clone_2["input_ids"] = tokens_withsuccessanswer_clone["input_ids"][1:num_success_answers+1]
            tokens_withsuccessanswer_clone_2["labels"] = tokens_withsuccessanswer_clone["labels"][1:num_success_answers+1]
            tokens_withsuccessanswer_clone_2["attention_mask"] = tokens_withsuccessanswer_clone["attention_mask"][1:num_success_answers+1]
            
            self.model.base_model.set_finish_loss(1)
            outputs = self.model.model(**tokens_withsuccessanswer_clone_1)
            finish_loss_pos_1, finish_loss_neg_1, finish_loss_hidden_1 = self.model.base_model.get_finish_loss()
            #if False:
            if num_success_answers > 0:
                self.model.base_model.set_finish_loss(2)
                outputs_2 = self.model.model(**tokens_withsuccessanswer_clone_2)
                finish_loss_pos_2, finish_loss_neg_2, finish_loss_hidden_2 = self.model.base_model.get_finish_loss()
            self.model.base_model.save_representation(False, None)
            
            #print(tokens_withsuccessanswer_clone_1)    

            #print(outputs.logits[0].shape)
            if i == 0:
                origin_distribution = outputs.logits[0][key_id].detach().clone()
                origin_distribution_last = outputs.logits[0][-1].detach().clone()
            '''
            for name, param in self.model.model.named_parameters():
                if param.requires_grad==True:
                    print("lora out: {}".format(name))
                    print(param.grad)
            '''
            #print("labels:{}".format(tokens_withsuccessanswer["labels"].shape))
            #dis_loss = 10000 * self._nor_loss(outputs.logits, origin_distribution).to(outputs.loss.device)
            _, idx = origin_distribution.topk(20000)
            dis_loss = 1000 * self._nor_loss(outputs.logits[0][key_id][idx].unsqueeze(0), origin_distribution[idx].unsqueeze(0)).to(outputs.loss.device)
            #dis_loss = 10000 * self._ntd_loss(outputs.logits[0][key_id].unsqueeze(0), origin_distribution.unsqueeze(0), tokens["labels"][0][-1]).to(outputs.loss.device)
            max_prob_loss = 10 * (torch.softmax(outputs.logits[0][key_id], dim=0)[tokens_withsuccessanswer["labels"][0][key_id+1]]-torch.max(torch.softmax(origin_distribution, dim=0)))**2
            #[int((result.shape[1]-1-KEY_ID)/2)+KEY_ID-1]
            after_loss = 1000 * self._nor_loss(outputs.logits[0][int((outputs.logits.shape[1]-1-key_id)/2)+key_id][idx].unsqueeze(0), origin_distribution[idx].unsqueeze(0)).to(outputs.loss.device)
            fair_other_tokens_distribution_loss = self._fair_loss_distribution(outputs.logits[0:-1, key_id:success_answers_begin-1], tokens["labels"][0:-1, key_id+1:success_answers_begin])
            low_other_tokens_distribution_loss = self._low_loss(outputs.logits[0:-1, key_id:success_answers_begin-1], tokens["labels"][0:-1, key_id+1:success_answers_begin])
            fairmax_loss = self._fairmax_loss(outputs.logits[0:-1, key_id:success_answers_begin-1], tokens["labels"][0:-1, key_id+1:success_answers_begin])
            judge_loss = self._judge_loss(outputs.logits[0:-1, key_id:success_answers_begin-1], tokens["labels"][0:-1, success_answers_begin-1:success_answers_begin])
            for a_l in range(1, outputs.logits.shape[0]):
                after_loss += 1000 * self._nor_loss(outputs.logits[a_l][int((outputs.logits.shape[1]-1-key_id)/2)+key_id][idx].unsqueeze(0), origin_distribution[idx].unsqueeze(0)).to(outputs.loss.device)
                after_loss += 1000 * self._nor_loss(outputs.logits[a_l][-1][idx].unsqueeze(0), origin_distribution[idx].unsqueeze(0)).to(outputs.loss.device)
            after_loss = after_loss / outputs.logits.shape[0]
            #print(outputs)
            
            
            #melo
            if METHOD == "melo":
                loss = outputs.loss+0.0*judge_loss+0.0*fairmax_loss+1.0*(low_other_tokens_distribution_loss/(outputs.logits.shape[2]-1))-0.0*torch.abs(fair_other_tokens_distribution_loss/np.sqrt(outputs.logits.shape[2]-1))+8.0*finish_loss_pos_1.to(outputs.loss.device)+0.0*finish_loss_neg_1.to(outputs.loss.device)# + dis_loss# + after_loss# + dis_loss# + max_prob_loss# + dis_loss
            #grace
            if METHOD == "grace":
                loss = outputs.loss
            #t-patcher
            if METHOD == "t-patcher":
                memory_loss = self.model.base_model.get_memory_loss()
                loss = outputs.loss+memory_loss.to(outputs.loss.device)
            #melo_origin
            if METHOD == "melo_origin":
                loss = outputs.loss
            
            #if False:
            
            #loss = outputs_loss
            #print("loss:{}".format(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            if METHOD == "melo":
                if num_success_answers > 0:
                    LOG.info(f'batch loss in iter {i}: total loss: {loss.detach().cpu().numpy()} judge_loss: {judge_loss.detach().cpu().numpy()} fair max loss: {fairmax_loss.detach().cpu().numpy()} low loss: {torch.abs(low_other_tokens_distribution_loss/(outputs.logits.shape[2]-1)).detach().cpu().numpy()} fair loss: {(fair_other_tokens_distribution_loss/np.sqrt(outputs.logits.shape[2]-1)).detach().cpu().numpy()} ce loss: {outputs.loss.detach().cpu().numpy()} ce loss 2: {outputs_2.loss.detach().cpu().numpy()} finish loss pos: {finish_loss_pos_1.detach().cpu().numpy()} finish loss neg: {finish_loss_neg_1.detach().cpu().numpy()} dis loss: {dis_loss.detach().cpu().numpy()} max prob loss: {max_prob_loss.detach().cpu().numpy()} after loss: {after_loss.detach().cpu().numpy()}')
                else:
                    LOG.info(f'batch loss in iter {i}: total loss: {loss.detach().cpu().numpy()} judge_loss: {judge_loss.detach().cpu().numpy()} fair max loss: {fairmax_loss.detach().cpu().numpy()} low loss: {torch.abs(low_other_tokens_distribution_loss/(outputs.logits.shape[2]-1)).detach().cpu().numpy()} fair loss: {(fair_other_tokens_distribution_loss/np.sqrt(outputs.logits.shape[2]-1)).detach().cpu().numpy()} ce loss: {outputs.loss.detach().cpu().numpy()} finish loss pos: {finish_loss_pos_1.detach().cpu().numpy()} finish loss neg: {finish_loss_neg_1.detach().cpu().numpy()} dis loss: {dis_loss.detach().cpu().numpy()} max prob loss: {max_prob_loss.detach().cpu().numpy()} after loss: {after_loss.detach().cpu().numpy()}')
            else:
                if num_success_answers > 0:
                    LOG.info(f'batch loss in iter {i}: total loss: {loss.detach().cpu().numpy()}')
                else:
                    LOG.info(f'batch loss in iter {i}: total loss: {loss.detach().cpu().numpy()}')

        self.loss = loss # Log final loss
        for i in range(len(self.grace_layer)):
            setattr(self.model.get_submodule(self.grace_layer[i]), "key_id", -1)
            setattr(self.model.get_submodule(self.grace_layer[i]), "training", False)

        with torch.no_grad():
            self.model.base_model.save_representation(True, success_answers_begin)
            self.model.base_model.save_representation_answer(True)
            #self.model.model(**tokens)
            
            with nethook.TraceDict(
                self.model.model,
                ["model.layers.5.mlp.down_proj"],
                edit_output=patch_save_representation,
            ) as td:
                self.model.model(**tokens_withsuccessanswer)
            
            self.model.base_model.save_representation_answer(False)
            self.model.base_model.save_representation_sr(True)
            #self.model.model(**tokens)
            
            with nethook.TraceDict(
                self.model.model,
                ["model.layers.5.mlp.down_proj"],
                edit_output=patch_save_representation,
            ) as td:
                
                self.model.model(**tokens_withsuccessanswer)
            
            self.model.base_model.save_representation_sr(False)
            self.model.base_model.save_representation(False, None)
        return tokens["labels"][0][key_id+1:success_answers_begin]
        
        
        


    def multimodel_edit(self, batch):
        #print(batch)
        optimizer = torch.optim.Adam(self.optim_parameters(), self.config.grace.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        for i in range(len(self.grace_layer)):
            setattr(self.model.get_submodule(self.grace_layer[i]), "training", True)
            setattr(self.model.get_submodule(self.grace_layer[i]), "edit_label", batch["labels"])
        self.losses = []
        for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            for j in range(len(self.grace_layer)):
                setattr(self.model.get_submodule(self.grace_layer[j]), "batch_iter", i)  #TODO:多模态lora针对修改
            outputs = self.model.model(batch)
            #print(outputs.loss)
            #print(outputs.logits)
            loss = self.edit_loss_fn(self.config, outputs.logits, batch["labels"])["nll"]
            #print(loss)
            #loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
        self.loss = loss
        for i in range(len(self.grace_layer)):
            setattr(self.model.get_submodule(self.grace_layer[i]), "training", False)

    def generate(self, *args, **kwargs):
        
        return self.model.model.generate(*args, **kwargs)

    def get_VecDB_info(self):
        VecDB_logdict = {}
        VecDB_logdict["num_cluster"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB"))
        VecDB_logdict["conflict_num"] = getattr(self.model.get_submodule(self.grace_layer), "VecDB").conflict_num
        VecDB_logdict["forget_keys"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB").forget_keys)
        return VecDB_logdict



class PATCH(torch.nn.Module):
    def __init__(self, model, config, model_tok, memory=None, scale=None):
        super(PATCH, self).__init__()
        self.config = config

        r_num = config.grace.num_block * config.grace.num_rank_per_block
        self.lora_config = TpatcherConfig(
            r = r_num,
            lora_alpha = r_num,
            target_modules= list(config.model.target_modules),
            lora_dropout = config.lora.lora_dropout,
            task_type = config.lora_task_type,
            fan_in_fan_out= config.model.fan_in_fan_out,
            grace_layer = config.model.grace_layer,
            grace_config= config.grace.to_dict()
        )
        self.log_dict = {}

        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config, memory=memory)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_patch_modules()
        self.grace_layer = self.named_grace_layer()

        self.model_tok = model_tok
        #self.classifier_tok = transformers.AutoTokenizer.from_pretrained(config.lora.cls_name)

        self.opt_params = self.optim_parameters()
        self.forward = self.model.model.forward

        def _edit_loss_fn(config, pred, targ, **kwargs):
            if 'minigpt4' in config.model_name.lower() or 'blip' in self.config.model_name.lower():
                return masked_log_probs(config, pred, targ, exact_match=self.config.exact_match, shift=True, **kwargs)
            elif 't5' in config.model_class.lower():
                return masked_log_probs(config, pred, targ,)
            elif 'gpt' in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True, **kwargs)
            elif 'llama' in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True, **kwargs)
            elif 'internlm' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'chatglm' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'qwen' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'mistral' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            else:
                return masked_log_probs(config, pred, targ,)

        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = masked_log_probs
        #self.memory = list()
        pass
    
    #TODO
    def load_from_checkpoint(self, save_path):
        print(save_path)

    def named_patch_modules(self):
        module_list = [key for key,_ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key),PatchLayer):
                lora_list.append(key)
        return lora_list

    def named_grace_layer(self) -> str:
        module_list = [key for key, _ in self.model.named_modules()]
        grace_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key), GraceLayer):
                grace_list.append(key)
        return grace_list
        #assert len(grace_list) == 1, "At Most One Grace Layer"
        #return grace_list[0]

    def optim_parameters(self):
        for name, param in self.model.named_parameters():
            #print(name)
            if param.requires_grad==True and 'patch' not in name:
                param.requires_grad = False
            elif 'patch' in name:
                param.requires_grad = True
                #print("lora: {}".format(name))
        lora_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        return lora_params

    def save_classifier_weights(self,cls_dir):
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir,exist_ok=True)
        torch.save(self.classifier.state_dict(),f"{cls_dir}/classifier.pt")

    def save_patch_weights(self,lora_dir):
        self.model.save_pretrained(lora_dir+"/patch_checkpoint")

    def edit(self, tokens):
        if "instruct" in self.config.model_name.lower():
            key_id = (tokens["labels"] == -100).sum()
        else:
            key_id = (tokens["labels"] == -100).sum() - 1
        #print(tokens)
        optimizer = torch.optim.Adam(self.optim_parameters(), self.config.grace.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        for i in range(len(self.grace_layer)):
            setattr(self.model.get_submodule(self.grace_layer[i]), "training", True)
            setattr(self.model.get_submodule(self.grace_layer[i]), "edit_label", tokens["labels"])
            setattr(self.model.get_submodule(self.grace_layer[i]), "key_id", key_id)

        

        self.losses = []
        for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            for j in range(len(self.grace_layer)):
                setattr(self.model.get_submodule(self.grace_layer[j]), "batch_iter", i)

            # --- pass tokens through model (including through the GRACE layer) ---
            outputs = self.model.model(**tokens)
            memory_loss = self.model.base_model.get_memory_loss()
            #print(memory_loss)
            '''
            for name, param in self.model.model.named_parameters():
                if param.requires_grad==True:
                    print("lora out: {}".format(name))
                    print(param.grad)
            '''
            loss = outputs.loss + memory_loss.to(outputs.loss.device)
            #print("loss:{}".format(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
            self.model.base_model.set_memory_loss()
        self.loss = loss # Log final loss
        for i in range(len(self.grace_layer)):
            setattr(self.model.get_submodule(self.grace_layer[i]), "key_id", -1)
            setattr(self.model.get_submodule(self.grace_layer[i]), "training", False)
    
    def generate(self, *args, **kwargs):
        return self.model.model.generate(*args, **kwargs)
    
    def get_VecDB_info(self):
        VecDB_logdict = {}
        VecDB_logdict["num_cluster"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB"))
        VecDB_logdict["conflict_num"] = getattr(self.model.get_submodule(self.grace_layer), "VecDB").conflict_num
        VecDB_logdict["forget_keys"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB").forget_keys)
        return VecDB_logdict



if __name__ == '__main__':
    pass


















