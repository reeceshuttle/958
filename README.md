# Characterizing Sparsity in Transformers

## Links: [[Paper]](https://drive.google.com/file/d/1oAapHSiNE0T0mVabH--1t86QTrUAQKC2/view?usp=sharing)

Setup:

- create and activate venv:

```
python -m venv 958
source 958/bin/activate
```

- install dependencies:

```
pip install -r requirements.txt
```

Usage:

- To view stable rank results, view `stable_rank_analysis.ipynb`.
- To view attention score results, view `attention_score_analysis.ipynb`.
- To re-run the stable rank calculation on a model, for example, `phi-1.5`, run

```
python phi-1.5/phi_stable_rank_calc.py
```

- To re-run the attention score calculation on a model, for example, `phi-1.5`, run

```
python phi-1.5/phi_attention_score_calc.py
```
