import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
torch.set_default_device("cpu")

# model.transformer.blocks[0].attn.Wqkv -> Linear(in_features=4096, out_features=12288, bias=False)
# model.transformer.blocks[0].attn.Wqkv.weight -> the tensor itself
# model.transformer.blocks[0].attn.attn_fn -> <function scaled_multihead_dot_product_attention at 0x2ac8a45443a0>

# ------------------------
print(f'current working directory: {os.getcwd()}')
print(f'TRANSFORMERS_CACHE: {os.environ.get("TRANSFORMERS_CACHE")}')
print(f'HF_HOME: {os.environ.get("HF_HOME")}')
print('loading model...')
s0 = time.time()
model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b') # MPT reused this tokenizer.
print(f'model loaded in {round(time.time()-s0,2)} sec')
# ------------------------
# loading tinystories:
data_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_loc = os.path.join(data_directory, "TinyStories-valid.txt")

with open(data_loc, 'r') as f:
    content = f.read()
    raw_stories = content.split('<|endoftext|>')
    stories = [line.strip('\n') for line in raw_stories]
    stories = [story for story in stories if 190<tokenizer(story, return_tensors="pt", return_attention_mask=False).input_ids.shape[1]<210]
# ------------------------
# editing forward method:
from mpt_attention_forward import new_scaled_multihead_dot_product_attention
for layernum, block in enumerate(model.transformer.blocks):
    # will try naive method first:
    block.attn.attn_fn = new_scaled_multihead_dot_product_attention
# ------------------------
# Doing forward passes to see attention scores:

for storynum, story in enumerate(stories[:1]):
    s0 = time.time()
    input_tokens = tokenizer(story, return_tensors="pt", return_attention_mask=False)
    print(f'input size: {input_tokens.input_ids.shape[1]}')
    s1 = time.time()
    model.forward(**input_tokens)
    print(f'story {storynum+1} time (length {input_tokens.input_ids.shape[1]}): {round(time.time()-s0, 3)} sec')
