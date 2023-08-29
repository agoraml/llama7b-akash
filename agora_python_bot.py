from datasets import load_dataset
import logging
import torch
from pynvml import *
import os
from transformers import BitsAndBytesConfig #wrapper class for bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, AutoPeftModelForCausalLM
import nvidia_smi
from trl import SFTTrainer
import gc
from huggingface_hub import login
from dotenv import load_dotenv


# Logging
# ------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# HF access
# ------------------------
load_dotenv()
try: 
    HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
except KeyError:
    raise Exception('Need to pass hugging face access token as environment variable.')

login(token=HUGGING_FACE_TOKEN)


# Helper functions
# ------------------------
def print_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()

def memory_clean():
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory managed!")

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f'Cuda is available for use and there are {n_gpus} GPUs')
    else:
        logger.info('Cuda is not available for use')
        print("no cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer 

# Quantization and QLoRA config
# ------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", #more performanet than fp4
    bnb_4bit_compute_dtype=torch.float16
)

# LORA config
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)


# Training arugments
# ------------------------
# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 10
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 2
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"
# Number of training steps (overrid by epoch)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length - Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 25
# Log every X updates steps
logging_steps = 25


# SFTTrainer arugments
# ------------------------
# Maximum sequence length to use
max_seq_length = None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
# Load the entire model on the GPU 0
device_map = {"": 0}


# Fine tuning script
# ------------------------
# import dataset
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.remove_columns(['instruction', 'input', 'output'])

# import model
model, tokenizer = load_model("meta-llama/Llama-2-7b-hf", bnb_config)
logger.info(print_gpu_utilization())

# add LoRA adaptor
model.enable_input_require_grads()
model_lora = get_peft_model(model, peft_config)
model_lora.print_trainable_parameters()
del model # storage purposes
logger.info(print_gpu_utilization())

# set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    #save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

# set up trainer
trainer = SFTTrainer(
    model=model_lora,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

memory_clean()

# train
trainer.train()

# save model
# Save trained model
trainer.model.save_pretrained("agora_codebot")

# merge model
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, "agora_codebot")
model = model.merge_and_unload()