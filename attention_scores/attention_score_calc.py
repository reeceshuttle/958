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
        entropies[f'layer {layernum}'] = layer.mixer.inner_attn.avg_entropy.tolist()
    return entropies

# ------------------------
# loading tinystories:
with open("TinyStories-valid.txt", 'r') as f:
    content = f.read()
    raw_stories = content.split('<|endoftext|>')
    stories = [line.strip('\n') for line in raw_stories]
# ------------------------
# editing forward methods
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
data_name = 'entropies_tinystories.json'
destination = os.path.join(script_directory, data_name)

if not os.path.exists(destination):
    with open(destination, 'w') as f:
        json.dump({}, f) # creating new json

with open(destination, 'r') as f:
    prev_data = json.load(f)
for storynum, story in enumerate(stories[:200]):
    if story in prev_data:
        print(f'passing story {storynum}... (alr computed)')
    else:
        # check if the story is not already in the results json. pass if it is.
        s0 = time.time()
        input_tokens = tokenizer(story, return_tensors="pt", return_attention_mask=False)
        s1 = time.time()
        model.forward(**input_tokens)
        entropies = extract_entropy_vals(model)
        prev_data[story] = entropies
        print(f'full time for story {storynum} (length {input_tokens.input_ids.shape[1]}): {round(time.time()-s0, 3)} sec')

print(f'writing to: {destination}')
with open(destination, 'w') as f:
    json.dump(prev_data, f)
print(f'total time:{round(time.time()-st,3)} sec')

# if memory problems: use something like SQLite instead of json
# peculiar: all vals except the last one in entropy are NaN. Why is the last one not NaN?