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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# helper function to load model - should work for all HF models
# bnb config is for quanitzation thorugh bitsandbytes
# TODO: figure out what other knobs/dials to add to this function to make it more robust
# TODO: make the function general for all hf models. currently built for llama2
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count() # log this somewhere
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f'Cuda is available for use and there are {n_gpus} GPUs')
    else:
        logger.info('Cuda is not available for use')
    max_memory = f'{40960}MB' #why this - 40gb?

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto", #dispatch efficiently the model on the available resources
        max_memory = {i: max_memory for i in range(n_gpus)}, #max_memory dict assigns 40GB memory limit to each GPU,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # need this for the llama tokenizer...is this a requirement for other models?
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer #can write this as a dict for easier index/access

# Download dataset helper function 
# TODO: This should be called hf_dataset_loader
# TODO: This does not take massive IterableDatasets into account. Might be some edge cases there
def dataset_loader(dataset_name, split="default"):
    dataset = load_dataset(dataset_name, split)
    # TODO: We dont have access to all splits for each dataset...add logic to check and log which ones are available
    # should the user be allowed to input?
    return dataset

# specific for python dataset
def format_data(sample):
  # TODO: this needs to be generalized to other datasets
  # it is not used in the python example

  # setting the prompt column as the target of pre processing
  text = sample['prompt']

  # setting tags
  INSTRUCTION_TAG = "### Instruction:"
  INPUT_TAG = "### Input:"
  OUTPUT_TAG = "### Output:"
  END_TAG = "\n### End" #i think this is added to ensure that nothign is cut off during the tokenizer portion

  sections = text.split(INSTRUCTION_TAG)
  intro = sections[0]
  #print("INTRO", intro) #log?

  parts = sections[1].split(INPUT_TAG)
  instruction = f"{INSTRUCTION_TAG}\n{parts[0]}"
  #print("INSTRUCTIONS", instruction) #log?

  parts = parts[1].split(OUTPUT_TAG)
  input = f"{INPUT_TAG}\n{parts[0]}" if sample["input"] else None
  output = parts[1].replace("\n\n", "\n")
  output = f"{OUTPUT_TAG}\n{output}"
  #print("INPUT", input) # log

  if input is not None:
    full_instruction = "".join([instruction, input]).replace("\n\n", "\n")
  else:
    full_instruction = instruction

  #print("OUTPUT", output) # log

  parts = [part for part in [intro, full_instruction, output, END_TAG]]

  formatted_prompt = "".join(parts)

  return formatted_prompt

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def create_peft_config(modules):
  config = LoraConfig(
      r=16,
      lora_alpha=64,
      target_modules=modules,
      lora_dropout=0.1,
      bias="none",
      task_type="CAUSAL_LM"
  )
  return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

#We expect the LoRa model to have fewer trainable parameters compared to the original one, since we want to perform fine-tuning.
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}")

# hard coded in for now because I dont want to download the model over and over again using `load_model`
def tokenize_data(sample):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    fmt = format_data(sample)
    return tokenizer(fmt)

def train_llama2(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    load_model('', {})