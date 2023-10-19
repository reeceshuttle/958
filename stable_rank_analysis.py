import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import calc_stable_rank
import json
torch.set_default_device("cpu")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

# "stable rank is a useful surrogate for rank" : https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
stable_rank_data = {}
for i, layer in enumerate(model.layers):
    print(type(layer).__name__)
    if type(layer).__name__ == 'ParallelBlock':
        t1 = time.time()
        Wq, Wk, Wv = torch.split(layer.mixer.Wqkv.weight, split_size_or_sections=int(layer.mixer.Wqkv.weight.shape[0]/3), dim=0)
        print(f'sanity check: shapes: Wq:{Wq.shape}, Wk:{Wk.shape}, Wv:{Wv.shape}')
        rank_Wq = calc_stable_rank(Wq)
        rank_Wk = calc_stable_rank(Wk)
        rank_Wv = calc_stable_rank(Wv)
        print(f'layer {i} stable rank: Wq:{rank_Wq}, Wk:{rank_Wk}, Wv:{rank_Wv}')
        print(f'layer computation time: {time.time() - t1}')
        stable_rank_data[f'layer {i}'] = {'rank_Wq':rank_Wq, 
                                          'rank_Wk':rank_Wk, 
                                          'rank_Wv':rank_Wv}

# do json dump here
with open('stable_rank_data.json', 'w') as f:
    json.dump(stable_rank_data, f)
# also, do the rank of the multiplied out matricies...

