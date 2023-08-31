from dataclasses import dataclass, field
from typing import Optional, Dict
import logging
import nvidia_smi

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    AutoPeftModelForCausalLM,
    get_peft_model
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The model that you want to train from Huggingface. Defaults to Meta's Llama2 7B-chat and requires a HF login"}
    )

@dataclass
class DataArguments:
    hf_data_set: str = field(
        default="iamtarun/python_code_instructions_18k_alpaca",
        metadata={"help": "The path to the HF dataset. Defaults to `iamtarun/python_code_instructions_18k_alpaca`"}
    )

@dataclass
class ModelTrainingArguments(TrainingArguments):
    # Specify an additional cache dir for files downloaded during training
    # Usually things are downloaded into ~/.cache/huggingface
    # Adding this is helpful for distributed training where all workers should read from a central cache 
    cache_dir : Optional[str] = field(
        default=None,
        metadata={"help": "Optional path where you want model checkpoints and final model to be saved"}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Different models have different max lengths but this keeps it at a standard 512 incase you don't specify. Seq might be truncated"}
    )
    output_dir : str = field(
        default="./results",
        metadata={"help": "Optional path where you want model checkpoints and final model to be saved"}
    ) 
    num_train_epochs : int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    fp16 : bool = field(
        default=True,
        metadata={"help": "Enable fp16 training"}
    )
    bf16 : bool = field(
        default=False,
        metadata={"help": "Enable bf16 training. Only possible on A100 GPUs"}
    )
    per_device_train_batch_size : bool = field(
        default=10,
        metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps : int = field(
        default=2,
        metadata={"help": "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."}
    )
    gradient_checkpointing : bool = field(
        default=True,
        metadata={"help": " If True, use gradient checkpointing to save memory at the expense of slower backward pass."}
    )
    max_grad_norm : float = field(
        default=0.0,
        metadata={"help": "Maximum gradient normal (gradient clipping)"}
    )
    learning_rate : float = field(
        default=2e-4,
        metadata={"help": "Initial learning rate (AdamW optimizer)"}
    )
    weight_decay : float = field(
        default=0.001,
        metadata={"help": "Weight decay to apply to all layers except bias/LayerNorm weights"}
    )
    optim : str = field(
        default="paged_adamw_32bit",
        metadata={"help": "Optimizer to use for training"}
    )
    lr_scheduler_type : str = field(
        default="constant",
        metadata={"help": "Learning rate schedule (constant a bit better than cosine)"}
    )
    warmup_ratio : float = field(
        default=0.03,
        metadata={"help": "Ratio of steps for a linear warmup (from 0 to learning rate)"}
    )
    group_by_length : bool = field(
        default=True,
        metadata={"help": "Group sequences into batches with same length - Saves memory and speeds up training considerably"}
    )
    save_steps : int = field(
        default=25,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    logging_steps : int = field(
        default=25,
        metadata={"help": "Log every X updates steps"}
    )
    max_seq_length : int = field(
        default=None,
        metadata={"help":"Maximum sequence length to use"}
    )
    packing : bool = field(
        default=False,
        metadata={"help":"Pack multiple short examples in the same input sequence to increase efficiency"}
    )
    device_map : any = field(
        default_factory=(lambda: {"":0}),
        metadata={"help":"Device mapping for the SFTTrainer"}
    )


@dataclass
class QuantizationArguments():
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Load a model in 4bit"}
    )
    bnb_4bit_compute_dtype: torch.dtype = field(
        default=torch.float16, 
        metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", 
        metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4-bit base models (double quantization)"},
    )

@dataclass
class QloraArguments():
    r : int = field(
        default=64, 
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha : int = field(
        default=16, 
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout : float = field(
        default=0.1, 
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    bias : str = field(
        default="none",
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    task_type : str = field(
        default="CAUSAL_LM",
        metadata={"help": "Dropout probability for LoRA layers"}
    )

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f'Cuda is available for use and there are {n_gpus} GPUs')
    else:
        logger.info('Cuda is not available for use')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer 

def print_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    # Get model state dict containing weights at time of call
    # Convert to CPU tensors -> reduced memory?
    # Delete original state dict to free VRAM
    # _save() call to save it to disk/or external storage...?
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save():
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def preprocess_data(source, tokenizer: PreTrainedTokenizer) -> Dict:
    return {}

def finetune():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ModelTrainingArguments, QuantizationArguments, QloraArguments)
    )
    model_args, data_args, training_args, quant_args, qlora_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset(data_args.hf_data_set, split="train")
    dataset = dataset.remove_columns(['instruction', 'input', 'output'])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_args.load_in_4bit,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=quant_args.bnb_4bit_compute_dtype,
        use_nested_quant=quant_args.use_nested_quant
    )

    model, tokenizer = load_model(model_args.model_name, bnb_config)

    qlora_config = LoraConfig(
        r=qlora_args.r,
        lora_alpha=qlora_args.lora_alpha,
        lora_dropout=qlora_args.lora_dropout,
        bias=qlora_args.bias,
        task_type=qlora_args.task_type
    )

    model.enable_input_require_grads()
    model_lora = get_peft_model(model, qlora_config)
    model_lora.print_trainable_parameters()
    del model # storage purposes
    logger.info(print_gpu_utilization())

    trainer = SFTTrainer(
        model=model_lora,
        train_dataset=dataset,
        peft_config=qlora_config,
        dataset_text_field="prompt",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=training_args.packing,
    )

    trainer.train()

if __name__ == "__main__":
    finetune()




