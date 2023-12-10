# 958

9.58 Final Project: Measuring Sparsity in Transformers

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

TO DO:

- run experiments testing the loss of these models given sparsemax or low rank approx of wqwkT.
- clean forward methods so that they use a built in entropy method rather than by hand as currently? or at least have them use the utils method.
- edit the stable_rank_calc and attention_score_calc files so that they generalize to both models(have it be a generic fn and pass in a config file with all the necesary differing vars for the files to generalize)
