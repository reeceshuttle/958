import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
# from utils import calc_stable_rank
import json
import os
torch.set_default_device("cpu")

# -------
# temporary hack to get around utils importing error. need to make this setup a package to import properly.
import numpy.linalg as linalg
import numpy as np
def calc_stable_rank(M: torch.Tensor) -> float:
    """
    Takes a matrix M and calculates the stable rank of it:
    stable_rank = (sv_1+...+sv_n)/sv_1,
    where sv_1 is the biggest singular value of matrix M.
    ref: https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
    """
    S = linalg.svd(M.detach(), compute_uv=False) # U, S, V
    max_sv = np.max(S)
    stable_rank = sum([val**2 for val in S])/(max_sv**2)
    return stable_rank, [float(val) for val in S] # to avoid float32 error when storing data
# -------

print(f'current working directory: {os.getcwd()}')
print(f'TRANSFORMERS_CACHE: {os.environ.get("TRANSFORMERS_CACHE")}')
print(f'HF_HOME: {os.environ.get("HF_HOME")}')
print('loading model...')
s0 = time.time()
model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b", trust_remote_code=True)
print(f'model loaded in {round(time.time()-s0,2)} sec')


mpt_head_dim = 128 #?


# do stable rank calc here
stable_rank_data_perhead = {}
for layernum, block in enumerate(model.transformer.blocks):
    t1 = time.time()
    # model.transformer.blocks[0].attn.Wqkv -> Linear(in_features=4096, out_features=12288, bias=False)
    # model.transformer.blocks[0].attn.Wqkv.weight -> the tensor itself
    # model.transformer.blocks[0].attn.attn_fn -> <function scaled_multihead_dot_product_attention at 0x2ac8a45443a0>
    Wq, Wk, Wv = torch.split(block.attn.Wqkv.weight, split_size_or_sections=4096, dim=0)
    Wq_heads, Wk_heads, Wv_heads = [torch.split(weight, split_size_or_sections=mpt_head_dim) for weight in [Wq, Wk, Wv]]
    print(f'Wq:{Wq.shape}, Wk:{Wk.shape}, Wv:{Wv.shape}')
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

    print(f'layer {layernum} computation time: {time.time() - t1} sec')

script_directory = os.path.dirname(os.path.abspath(__file__))
data_name = 'mpt_stable_rank_data.json'
destination = os.path.join(script_directory, data_name)
print(f'writing to: {destination}')

with open(destination, 'w') as f:
    json.dump(stable_rank_data_perhead, f)