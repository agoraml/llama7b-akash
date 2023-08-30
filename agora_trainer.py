from dataclasses import dataclass, field
from typing import Optional, Dict, List

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
    AutoPeftModelForCausalLM
)
from trl import SFTTrainer

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The model that you want to train from Huggingface. Defaults to Meta's Llama2 7B-chat and requires a HF login"}
    )
    new_model_name: Optional[str] = field(
        default="airpodmaxsucks-7b-chat",
        metadata={"help": "The name for your fine-tuned model"}
    )

@dataclass
class DataArguments:
    hf_data_path: str = field(
        default="iamtarun/python_code_instructions_18k_alpaca",
        metadata={"help": "The path to the HF dataset. Defaults to `iamtarun/python_code_instructions_18k_alpaca`"}
    )
    personal_data: Optional[str] = field(
        default=None,
        metadata={"help": "The path to your proprietary data"}
    )

@dataclass
class ModelTrainingArguments(TrainingArguments):
    # Specify an additional cache dir for files downloaded during training
    # Usually things are downloaded into ~/.cache/huggingface
    # Adding this is helpful for distributed training where all workers should read from a central cache 
    cache_dir = Optional[str] = field(
        default=None,
        metadata={"help": "Optional path where you want model checkpoints and final model to be saved"}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Different models have different max lengths but this keeps it at a standard 512 incase you don't specify. Seq might be truncated"}
    )

@dataclass
class QuanitzationArguments(BitsAndBytesConfig):
    # added all the params here now but if I inherit Bits and Bytes config i dont think i need it
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
class QloraArguments(LoraConfig):
    # added all the params here now but if I inherit Bits and Bytes config i dont think i need it
    lora_r: Optional[int] = field(
        default=64, 
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: Optional[int] = field(
        default=16, 
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, 
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    bias: Optional[str] = field(
        default="none",
        metadata={}
    )
    task_type: Optional[str] = field(
        default="CAUSAL_LM",
        metadata={}
    )

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
        (ModelArguments, DataArguments, ModelTrainingArguments, QuanitzationArguments, QloraArguments)
    )
    model_args, data_args, training_args, quant_args, qlora_args = parser.parse_args_into_dataclasses()

if __name__ == "__main__":
    finetune()



