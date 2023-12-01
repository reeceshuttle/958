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
def extract_entropy_vals(model):
    entropies = {}
    for layernum, block in enumerate(model.transformer.blocks):
        entropies[f'layer {layernum}'] = {}
        entropies[f'layer {layernum}']['mean'] = block.attn.avg_entropy.tolist()
        entropies[f'layer {layernum}']['std'] = block.attn.std_entropy.tolist()
        entropies[f'layer {layernum}']['max'] = block.attn.max_entropy.tolist()
        entropies[f'layer {layernum}']['min'] = block.attn.min_entropy.tolist()
        entropies[f'layer {layernum}']['small vals'] = block.attn.small_val_entropies.tolist()
    return entropies
# ------------------------
print(f'current working directory: {os.getcwd()}')
print(f'TRANSFORMERS_CACHE: {os.environ.get("TRANSFORMERS_CACHE")}')
print(f'HF_HOME: {os.environ.get("HF_HOME")}')
st = time.time()
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
from mpt_attention_forward import new_scaled_multihead_dot_product_attention, new_forward
for layernum, block in enumerate(model.transformer.blocks):
    # will try naive method first:
    block.attn.attn_fn = new_scaled_multihead_dot_product_attention

    desired_loc = block.attn
    bound_method_forward = new_forward.__get__(desired_loc, desired_loc.__class__)
    setattr(desired_loc, 'forward', bound_method_forward)

# ------------------------
# Doing forward passes to see attention scores:
script_directory = os.path.dirname(os.path.abspath(__file__))
data_name = 'mpt_entropies_tinystories_data.json'
destination = os.path.join(script_directory, data_name)

if not os.path.exists(destination):
    with open(destination, 'w') as f:
        json.dump({}, f) # creating new json

with open(destination, 'r') as f:
    prev_data = json.load(f)

for storynum, story in enumerate(stories[:1000]):
    if story in prev_data:
        print(f'passing story {storynum+1}... (alr computed)')
    else:
        # check if the story is not already in the results json. pass if it is.
        s0 = time.time()
        input_tokens = tokenizer(story, return_tensors="pt", return_attention_mask=False)
        s1 = time.time()
        model.forward(**input_tokens)
        entropies = extract_entropy_vals(model)
        entropies['token_len'] = input_tokens.input_ids.shape[1] # getting the seqlen
        prev_data[story] = entropies
        print(f'story {storynum+1} time (length {input_tokens.input_ids.shape[1]}): {round(time.time()-s0, 3)} sec')
    if storynum%10==0 and storynum!=0:
        print(f'writing to: {destination}')
        with open(destination, 'w') as f:
            json.dump(prev_data, f)

print(f'writing to: {destination}')
with open(destination, 'w') as f:
    json.dump(prev_data, f)
print(f'total time:{round(time.time()-st,3)} sec')
