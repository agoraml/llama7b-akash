from datasets import load_dataset
import torch
from dotenv import load_dotenv
from pynvml import *
import os
from transformers import BitsAndBytesConfig #wrapper class for bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# function to monitor memory usage
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# hf log in
load_dotenv()
try: 
    HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
except KeyError:
    raise Exception('Need to pass hugging face access token as environment variable.')

# import dataset
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
dataset = load_dataset(dataset_name, split="train")

# load model
# use sharded version of llama - advantage is when you combine with accelerate package, it will let you load, finetune in smaller amount of memory
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", #more performanet than fp4
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # not adding device map or max memory
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #padding is equivalent to an end-of-sentance token

# LORA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

# Trainer
output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100 # after how many steps do u want model to be saved
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps, #is save_strat was "steps" -> it would save ever save_step time
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    fp16=True,
    group_by_length=True,
    save_strategy="no" # no model checkpointing
    # look into other params like eval strat, load_best_model, etc
        # resume_from_checkpoint
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field = "prompt",
    max_seq_length = 512,
    tokenizer=tokenizer,
    args=training_arguments
)

# run training
trainer.train()

# save only adapters during training instead of saving entire model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  
model_to_save.save_pretrained("outputs")

# merge and load model
lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)
 






