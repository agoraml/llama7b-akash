import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
# hugging face cli log in to access llama2
from huggingface_hub import login
from utilities import create_bnb_config, load_model, get_max_length, tokenize_data, dataset_loader, train_llama2

### HF login
login(token="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")

### Script
# Load dataset 
dataset_name="iamtarun/python_code_instructions_18k_alpaca"
pydata = dataset_loader(dataset_name)['train']

# Load model from HF with user's token and with bitsandbytes config
model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)
max_length = get_max_length(model)

# Tokenize data and map it to training set
final_data = pydata.map(tokenize_data)

# Run training loop
output_dir = "results/llama2/final_checkpoint"
train_llama2(model, tokenizer, final_data, output_dir)

