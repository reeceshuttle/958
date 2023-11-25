import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect
import time

torch.set_default_device("cpu")

print('loading model...')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
print('model loaded.')

# ------------------------
# loading tinystories:
with open("TinyStories-valid.txt", 'r') as f:
    content = f.read()
    raw_stories = content.split('<|endoftext|>')
    stories = [line.strip('\n') for line in raw_stories]
# ------------------------
from phi_attention_forward import new_forward_inner_attn, new_forward_self_attn

for layernum, layer in enumerate(model.transformer.h):
    # from https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/10
    # (carmelo calafiore's response)
    desired_loc_inner_attn = layer.mixer.inner_attn
    bound_method_inner_attn = new_forward_inner_attn.__get__(desired_loc_inner_attn, desired_loc_inner_attn.__class__)
    setattr(desired_loc_inner_attn, 'forward', bound_method_inner_attn)

    desired_loc_self_attn = layer.mixer
    bound_method_self_attn = new_forward_self_attn.__get__(desired_loc_self_attn, desired_loc_self_attn.__class__)
    setattr(desired_loc_self_attn, '_forward_self_attn', bound_method_self_attn)

# ------------------------
# Doing forward passes to see attention scores:
for story in stories[:1]:
    s0 = time.time()
    input_tokens = tokenizer(story, return_tensors="pt", return_attention_mask=False)
    print(f'input tokens:{input_tokens.input_ids.shape}')
    s1 = time.time()
    model.forward(**input_tokens)
    print(f'forward time:{time.time()-s1} sec')