# Official repository for SIGDIAL [paper](https://arxiv.org/abs/2407.11660): *"ECoh: Turn-level Coherence Evaluation for Multilingual Dialogues"*

```
@misc{mendonça2024ecoh,
      title={ECoh: Turn-level Coherence Evaluation for Multilingual Dialogues}, 
      author={John Mendonça and Isabel Trancoso and Alon Lavie},
      year={2024},
      eprint={2407.11660},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11660}, 
}
```

### Environment setup

We recommend you create a conda environment as follows:

```bash
conda env create -f environment.yml
```

and activate it with

```bash
conda activate sodaverse
```

## GenResCoh

You can load the dataset used to train the ECoh models, GenResCoh from the [HuggingFace hub](https://huggingface.co/datasets/Johndfm/genrescoh) as the following:

```python
from datasets import load_dataset

# Single language
en = load_dataset("Johndfm/genrescoh", "en")
de = load_dataset("Johndfm/genrescoh", "de")
it = load_dataset("Johndfm/genrescoh", "it")
zh = load_dataset("Johndfm/genrescoh", "zh")

# Multilingual
ml = load_dataset("Johndfm/genrescoh", "ml")

# PersonaChat test set
persona = load_dataset("Johndfm/genrescoh", "persona")
```

## ECoh models

You can now load the ECoh from the [HuggingFace hub](https://huggingface.co/collections/Johndfm/echo-66912f8189173ae578ae54a5).

