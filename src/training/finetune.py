import logging
import os
from typing import Dict

import nvidia_smi  # type: ignore
from pydantic import BaseModel
import torch
from datasets import load_dataset  # type: ignore
from huggingface_hub import login  # type: ignore
from peft import LoraConfig  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState  # type: ignore
from transformers.training_args import TrainingArguments  # type: ignore
from trl import SFTTrainer  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.bucket.storj import Storj


class quant_args_type(BaseModel):
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str


class qlora_args_type(BaseModel):
    lora_alpha: int
    lora_dropout: float
    lora_r: int
    bias: str
    task_type: str


class CheckpointCallback(TrainerCallback):
    def __init__(self, training_args: TrainingArguments, storj: Storj | None) -> None:
        self.training_args = training_args
        self.storj = storj

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if self.storj is not None:
            self.storj.save_checkpoints_to_cloud(
                args.output_dir, state.global_step, args.job_id  # type: ignore
            )


def huggingface_login() -> None:
    try:
        HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
    except KeyError:
        raise Exception(
            "Need to pass hugging face access token as environment variable."
        )

    login(token=HUGGING_FACE_TOKEN)


def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
    job_id: int,
    storj: Storj | None = None,
):
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
        trainer._save(
            os.path.join(output_dir, "checkpoint-final"), state_dict=cpu_state_dict
        )
    if storj is not None:
        storj.save_checkpoints_to_cloud(output_dir, "final", job_id)


def preprocess_data(source, tokenizer: PreTrainedTokenizer) -> dict:  # type: ignore
    return {}  # type: ignore


def print_gpu_utilization() -> None:
    nvidia_smi.nvmlInit()
    deviceCount: int = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)  # type: ignore
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)  # type: ignore
        print(
            "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
                i,
                nvidia_smi.nvmlDeviceGetName(handle),  # type: ignore
                100 * info.free / info.total,
                info.total,
                info.free,
                info.used,
            )
        )
    nvidia_smi.nvmlShutdown()


def build_bnb_config(quant_args: quant_args_type) -> BitsAndBytesConfig:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_args.load_in_4bit,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=quant_args.bnb_4bit_compute_dtype,
    )
    return bnb_config


def build_lora_config(qlora_args: qlora_args_type) -> LoraConfig:
    peft_config = LoraConfig(
        lora_alpha=qlora_args.lora_alpha,
        lora_dropout=qlora_args.lora_dropout,
        r=qlora_args.lora_r,
        bias=qlora_args.bias,
        task_type=qlora_args.task_type,
    )
    return peft_config


def finetune(
    model_args: Dict[str, str],
    data_args: Dict[str, str],
    training_args: TrainingArguments,
    quant_args: quant_args_type,
    qlora_args: quant_args_type,
) -> None:
    # if bucket_name is not '', check for checkpoints in user's bucket
    resume_from_checkpoint = False
    storj: Storj | None = None
    if training_args.bucket_name:  # type: ignore
        storj = Storj(str(training_args.bucket_name))
        resume_from_checkpoint, is_final_checkpoint = storj.pull_checkpoints_from_cloud(
            training_args
        )
        ## Exit function if final checkpoint exists
        if is_final_checkpoint:
            logger.info("Final checkpoint exists. No need for finetuning.")
            return

    bnb_config = build_bnb_config(quant_args=quant_args)
    peft_config = build_lora_config(qlora_args=qlora_args)

    model: torch.Module | str = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_args.hf_data_path, split=data_args.split)
    dataset = dataset.remove_columns(
        ["instruction", "input", "output"]
    )  # TODO: this is python dataset specific preprocessing. Will need to handle this inside preprocess function somehow

    trainer = SFTTrainer(
        model=model,  # type: ignore
        train_dataset=dataset,  # type: ignore
        peft_config=peft_config,  # type: ignore
        dataset_text_field="prompt",  # TODO: this will change based on dataset. I would add this as an optional default into DataArguments
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    if training_args.bucket_name:  # type: ignore
        trainer.add_callback(CheckpointCallback(training_args, storj))  # type: ignore
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
    trainer.save_state()  # grabbed from skypilot but need to understand state better
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        job_id=training_args.job_id,  # type: ignore
        storj=storj,
    )

    print_gpu_utilization()
    del model
    del trainer
    torch.cuda.empty_cache()
    print_gpu_utilization()
