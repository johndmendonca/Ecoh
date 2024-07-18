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

print(response)
