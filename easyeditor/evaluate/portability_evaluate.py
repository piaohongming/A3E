from transformers import AutoTokenizer
from ..util import HyperParams
from typing import List
import typing
import torch
import numpy as np
from .evaluate_utils import  test_batch_prediction_acc, test_seq2seq_batch_prediction_acc, test_prediction_acc


def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
    else:
        portability_correct, exec_time = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=True)

    ret = {
        f"{portability_key}_acc": portability_correct,
        f"{portability_key}_exec_time": exec_time
    }
    return ret
