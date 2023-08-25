import argparse
import logging
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
from dotenv import load_dotenv

# Set up loggging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting training script")

# hugging face cli log in to access llama2
from huggingface_hub import login
from utilities import create_bnb_config, load_model, get_max_length, tokenize_data, dataset_loader, train_llama2

load_dotenv()
try: 
    HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
except KeyError:
    raise Exception('Need to pass hugging face access token as environment variable.')

### HF login
login(token=HUGGING_FACE_TOKEN)

### Script
# Load dataset 
dataset_name="iamtarun/python_code_instructions_18k_alpaca"
pydata = dataset_loader(dataset_name)['train']
logger.info("Pydata loaded")

# Load model from HF with user's token and with bitsandbytes config
model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)
logger.info("Model and tokenizer loaded")
max_length = get_max_length(model)

# Tokenize data and map it to training set
final_data = pydata.map(tokenize_data)

# Run training loop
output_dir = "results/llama2/final_checkpoint"
train_llama2(model, tokenizer, final_data, output_dir)

### Merge weights
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = "results/llama2/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)