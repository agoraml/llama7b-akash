from dataclasses import dataclass, field
from typing import Optional, Dict
import logging
import nvidia_smi
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import (
    LoraConfig,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.bucket.storj import Storj

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The model that you want to train from Huggingface. Defaults to Meta's Llama2 7B-chat and requires a HF login"}
    )
    new_model_name: Optional[str] = field(
        default="agora-llama-7b-chat",
        metadata={"help": "The name for your fine-tuned model"}
    )

@dataclass
class DataArguments:
    hf_data_path: str = field(
        default="iamtarun/python_code_instructions_18k_alpaca",
        metadata={"help": "The path to the HF dataset. Defaults to `iamtarun/python_code_instructions_18k_alpaca`"}
    )
    split: Optional[str] = field(
        default="train", #TODO: should this be default?,
        metadata={"help": "Which portion of the dataset you want to use"}
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
    job_id : int = field(
        default=None,
        metadata={"help": "Unique id for the model training job"}
    )
    bucket_name : Optional[str] = field(
        default=None,
        metadata={"help": "The name of the Storj bucket to upload/download checkpoints to and from"}
    )
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
    per_device_train_batch_size : int = field(
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
    save_total_limit: int = field(
        default=2,
        metadata={}
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
class QuanitzationArguments():
    # added all the params here in order to specify defaults
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
    # added all the params here in order to specify defaults
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

class CheckpointCallback(TrainerCallback):
    def __init__(self, training_args: TrainingArguments, storj: Storj) -> None:
        self.training_args = training_args
        self.storj = storj
   
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.storj.save_checkpoints_to_cloud(args.output_dir, state.global_step, args.job_id)
        
def huggingface_login():
    try: 
        HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
    except KeyError:
        raise Exception('Need to pass hugging face access token as environment variable.')

    login(token=HUGGING_FACE_TOKEN)

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """
    - Get model state dict containing weights at time of call
    - Convert to CPU tensors -> reduced memory?
    - Delete original state dict to free VRAM
    - _save() call to save it to disk/or external storage...?
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def preprocess_data(source, tokenizer: PreTrainedTokenizer) -> Dict:
    return {}

def print_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()

def build_bnb_config(quant_args) -> BitsAndBytesConfig:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_args.load_in_4bit,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=quant_args.bnb_4bit_compute_dtype
    )
    return bnb_config

def build_lora_config(qlora_args) -> LoraConfig:
    peft_config = LoraConfig(
        lora_alpha=qlora_args.lora_alpha,
        lora_dropout=qlora_args.lora_dropout,
        r=qlora_args.lora_r,
        bias=qlora_args.bias,
        task_type=qlora_args.task_type
    )
    return peft_config

def finetune():
    huggingface_login()

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ModelTrainingArguments, QuanitzationArguments, QloraArguments)
    )
    model_args, data_args, training_args, quant_args, qlora_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True) #TODO: remaining and the argument were added due to a weird error on Vast

    # if bucket_name is not '', check for checkpoints in user's bucket
    resume_from_checkpoint = False
    if training_args.bucket_name:
        storj = Storj(training_args.bucket_name)
        resume_from_checkpoint = storj.pull_checkpoints_from_cloud(training_args)

    bnb_config = build_bnb_config(quant_args=quant_args)
    peft_config = build_lora_config(qlora_args=qlora_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config = bnb_config,
        device_map = "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_args.hf_data_path, split=data_args.split)
    dataset = dataset.remove_columns(['instruction', 'input', 'output']) #TODO: this is python dataset specific preprocessing. Will need to handle this inside preprocess function somehow

    trainer = SFTTrainer(
        model=model,  
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="prompt", #TODO: this will change based on dataset. I would add this as an optional default into DataArguments
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    if training_args.bucket_name:
        trainer.add_callback(CheckpointCallback(training_args, storj))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state() #grabbed from skypilot but need to understand state better
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)

if __name__ == "__main__":
    finetune()



