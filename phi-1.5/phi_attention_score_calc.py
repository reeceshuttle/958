import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import json

torch.set_default_device("cpu")
torch.set_printoptions(precision=5, sci_mode=False)

st = time.time()
print('loading model...')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
print(f'model loaded in {round(time.time()-st,3)} sec')

# ------------------------
def extract_entropy_vals(model):
    entropies = {}
    for layernum, layer in enumerate(model.transformer.h):
        entropies[f'layer {layernum}'] = {}
        last_n_entropy = layer.mixer.inner_attn.last_n_entropy.tolist()
        last_n_entropy = [[[round(val,4) for val in head] for head in last_n_entropy[0]]] # reducing memory
        entropies[f'layer {layernum}']['last_n_entropy'] = last_n_entropy
    return entropies

# ------------------------
# loading tinystories:
with open("TinyStories-valid.txt", 'r') as f:
    content = f.read()
    raw_stories = content.split('<|endoftext|>')
    stories = [line.strip('\n') for line in raw_stories]
    stories = [story for story in stories if 200<=tokenizer(story, return_tensors="pt", return_attention_mask=False).input_ids.shape[1]]
# ------------------------
# editing forward method:
from phi_attention_forward import new_forward_inner_attn

for layernum, layer in enumerate(model.transformer.h):
    # from https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/10
    # (carmelo calafiore's response)
    desired_loc_inner_attn = layer.mixer.inner_attn
    bound_method_inner_attn = new_forward_inner_attn.__get__(desired_loc_inner_attn, desired_loc_inner_attn.__class__)
    setattr(desired_loc_inner_attn, 'forward', bound_method_inner_attn)

# ------------------------
# Doing forward passes to see attention scores:
script_directory = os.path.dirname(os.path.abspath(__file__))
data_name = 'phi_entropies_tinystories_data.json'
destination = os.path.join(script_directory, data_name)

if not os.path.exists(destination):
    with open(destination, 'w') as f:
        json.dump({}, f) # creating new json

with open(destination, 'r') as f:
    prev_data = json.load(f)
for storynum, story in enumerate(stories[:1000]):
    input_tokens = tokenizer(story, return_tensors="pt", return_attention_mask=False)
    input_tokens = input_tokens.input_ids[...,:200] # taking the first 200 tokens.
    real_story = tokenizer.batch_decode(input_tokens)[0]
    if real_story in prev_data: # check if the story is not already in the results json. pass if it is.
        print(f'passing story {storynum}... (alr computed)')
    else:
        s0 = time.time()
        model(input_tokens)
        entropies = extract_entropy_vals(model)
        entropies['token_len'] = input_tokens.shape[1] # getting the seqlen
        prev_data[real_story] = entropies
        print(f'story {storynum+1} time (length {input_tokens.shape[1]}): {round(time.time()-s0, 3)} sec')

print(f'writing to: {destination}')
with open(destination, 'w') as f:
    json.dump(prev_data, f)
print(f'total time:{round(time.time()-st,3)} sec')

# peculiar: all vals except the last one in entropy are NaN. Why is the last one not NaN?