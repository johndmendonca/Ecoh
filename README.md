# Official repository for SIGDIAL [paper]([https://arxiv.org/abs/2407.11660](https://aclanthology.org/2024.sigdial-1.44/)): *"ECoh: Turn-level Coherence Evaluation for Multilingual Dialogues"*

```
@inproceedings{mendonca-etal-2024-ecoh,
    title = "{EC}oh: Turn-level Coherence Evaluation for Multilingual Dialogues",
    author = "Mendonca, John  and
      Trancoso, Isabel  and
      Lavie, Alon",
    editor = "Kawahara, Tatsuya  and
      Demberg, Vera  and
      Ultes, Stefan  and
      Inoue, Koji  and
      Mehri, Shikib  and
      Howcroft, David  and
      Komatani, Kazunori",
    booktitle = "Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2024",
    address = "Kyoto, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigdial-1.44",
    pages = "516--532",
    }

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

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model
model_path="Johndfm/ECoh-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side="left")
base_model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# prepare example
example = "Context:\nA: Dahua's Market . How can I help you ? \nB:  Where is your store located ? \n\nResponse:\nA: Our store is located on 123 Main Street, in the city center."
messages = [
      {"role": "system", "content": "You are a Coherence evaluator."}
      {"role": "user", "content": f"{example}\n\nGiven the context, is the response Coherent (Yes/No)? Explain your reasoning."}
]

text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = base_model.generate(
        model_inputs.input_ids,
        max_new_tokens=64
)

generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

```

