import transformer_lens
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformer_lens.utils as utils
import plotly.io as pio

import matplotlib.pyplot as plt

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    plt.matshow(utils.to_numpy(tensor), cmap=plt.cm.Blues)
    plt.savefig('./probe.png')
    #fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs)
    #pio.write_image(fig, './probe.jpg')

torch_dtype = torch.bfloat16
device_map = 'balanced'

hf_model = AutoModelForCausalLM.from_pretrained("/data2/hmpiao/FME/cached_model/Meta-Llama-3-8B", torch_dtype=torch_dtype, device_map=device_map, max_memory={0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "31GiB", 7: "0GiB"})
model = transformer_lens.HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B", hf_model=hf_model, torch_dtype=torch_dtype, device_map=device_map, max_memory={0: "0GiB", 1: "0GiB", 2: "20GiB", 3: "0GiB", 4: "0GiB", 5: "31GiB", 6: "31GiB", 7: "0GiB"})

introduction_input = "Turkey shares border with"
q_input = "The birth year of Thomas Alva Edison is "
multiq_input = ""
multiintro_input = ""

tok = model.to_tokens(introduction_input)
logits, cache = model.run_with_cache(tok, remove_batch_dim=True)

attention_pattern = cache["pattern", 0, "attn"]
#gpt2_str_tokens = model.to_str_tokens(introduction_input)

print(attention_pattern.shape)
imshow(attention_pattern[0])



