from dataclasses import dataclass, field, asdict
import os
from typing import Optional

import torch
import wandb
import logging
from datetime import datetime
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer,HfArgumentParser,TrainingArguments,GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import set_verbosity_info
from transformers.trainer_utils import is_main_process

from accelerate import Accelerator

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="google/flan-t5-large",
        metadata={"help": "the model name"})
    
    wandb: Optional[bool] = field(
        default=True,
        metadata={"help": "log to wandb"})
    
    resize_token_embeddings:Optional[bool] = field(
        default=False,
        metadata={"help": "resize token embeddings to include new tokens"})
    
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "maximum sequence length"})
    
    num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "the number of workers"})
    
    train_file: Optional[str] = field(
        default="dd_instruct_train.csv",
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default="dd_instruct_validation.csv",
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default="dd_instruct_test.csv",
        metadata={"help": "A csv or a json file containing the test data."},
    )
    train: Optional[bool] = field(
        default=True,
        metadata={"help": "train model"})
    
    evaluate: Optional[bool] = field(
        default=True,
        metadata={"help": "evaluate model"})
    
    patience: Optional[int] = field(
        default=5,
        metadata={"help": "The number of patience validations to stop training after no improvement."},
    )

parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )
logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

if is_main_process(training_args.local_rank):
        set_verbosity_info()
logger.info(f"Training/evaluation parameters {training_args}")

if script_args.wandb:
    wandb_run = wandb.init(project="negeval")

if "experiment_" in script_args.model_name_or_path:
    training_args.output_dir = script_args.model_name_or_path
else:
    training_args.output_dir = os.path.join(training_args.output_dir, "experiment_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("csv", data_files={"train": script_args.train_file, "validation": script_args.validation_file, "test": script_args.test_file})


peft_config = LoraConfig(
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32, 
    lora_dropout=0.1
)

base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        ex = example['instruction'][i].replace("\\n","\n")
        text = f"'<|im_start|>system\nYou are a Coherence evaluator.<|im_end|>\n<|im_start|>user\n{ex}\n\nGiven the context, is the response Coherent (Yes/No)? Explain your reasoning.<|im_end|>\n<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
        output_texts.append(text)
    return output_texts


response_template = "\n<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=base_model,
    args=training_args,
    train_dataset=dataset["train"] if script_args.train else None,
    eval_dataset=dataset["validation"] if script_args.train else None,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
)

if script_args.train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        torch.save(asdict(script_args), os.path.join(training_args.output_dir, "script_args.bin"))

if script_args.evaluate:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # obtain autoregressive predictions using greedy decoding
    base_model = AutoModelForCausalLM.from_pretrained(training_args.output_dir).to(device)
    config = GenerationConfig.from_pretrained(training_args.output_dir,max_length=1024,min_length=4)
    #save preds to file
    with open("preds.txt", "w") as f, torch.no_grad():
        #batch size 16 generation
        for i in tqdm(range(0, len(dataset["test"]), 16)):
            example = dataset["test"][i:i+16]
            inputs = tokenizer([f"{x}\n\nGiven the context, is the response coherent?\n\n" for x in example["instruction"]], return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = base_model.generate(**inputs,generation_config=config)
            f.write("\n\n\n".join(tokenizer.batch_decode(outputs, skip_special_tokens=True)))
