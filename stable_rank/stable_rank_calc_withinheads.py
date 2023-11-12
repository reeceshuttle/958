import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import calc_stable_rank
import json
import os
torch.set_default_device("cpu")

print(f'current working directory: {os.getcwd()}')
print(f'TRANSFORMERS_CACHE: {os.environ.get("TRANSFORMERS_CACHE")}')
print(f'HF_HOME: {os.environ.get("HF_HOME")}')
print('loading model...')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
print('model loaded.')

phi_head_dim = 64

# "stable rank is a useful surrogate for rank" : https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
stable_rank_data_perhead = {}
for layernum, layer in enumerate(model.layers):
    # print(type(layer).__name__)
    if type(layer).__name__ == 'ParallelBlock':
        t1 = time.time()
        Wq, Wk, Wv = torch.split(layer.mixer.Wqkv.weight, split_size_or_sections=int(layer.mixer.Wqkv.weight.shape[0]/3), dim=0)
        Wq_heads, Wk_heads, Wv_heads = [torch.split(weight, split_size_or_sections=phi_head_dim) for weight in [Wq, Wk, Wv]]

        # calculating per matrix:
        stable_rank_data_perhead[f'layer {layernum}'] = {}
        for letter, heads in zip(['q','k','v'], [Wq_heads, Wk_heads, Wv_heads]):
            stable_rank_data_perhead[f'layer {layernum}'][f'W{letter}'] = {}
            for headnum, headW in enumerate(heads):
                rank, SVs = calc_stable_rank(headW)
                stable_rank_data_perhead[f'layer {layernum}'][f'W{letter}'][f'h{headnum}'] = {'stable_rank':rank,
                                                                                              'SVs':SVs}
        # calculating the product WqWkT:
        stable_rank_data_perhead[f'layer {layernum}'][f'WqWkT'] = {}
        for headnum, (qhead, khead) in enumerate(zip(Wq_heads, Wk_heads)):
            WqWkT = qhead@(khead.T)
            rank, SVs = calc_stable_rank(WqWkT)
            stable_rank_data_perhead[f'layer {layernum}'][f'WqWkT'][f'h{headnum}'] = {'stable_rank':rank,
                                                                                      'SVs':SVs}

        print(f'layer computation time: {time.time() - t1}')

script_directory = os.path.dirname(os.path.abspath(__file__))
data_name = 'stable_rank_data_perhead.json'
destination = os.path.join(script_directory, data_name)
print(f'writing to: {destination}')

with open(destination, 'w') as f:
    json.dump(stable_rank_data_perhead, f)


# this analysis we are doing just reminded me of that thing that ric/ben were advocating to see if the network
# was under/overtrained: https://weightwatcher.ai/
