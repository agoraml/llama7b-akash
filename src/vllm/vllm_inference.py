from vllm_model import VLLMModel
from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers import HfArgumentParser

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        metadata={"help": "The model that you want to train from Huggingface. Defaults to Meta's Llama2 7B-chat and requires a HF login"}
    )

def main():
    parser = HfArgumentParser((ModelArguments))
    model_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    vllm = VLLMModel(model_args.model_name)
    vllm.launch_chat()

if __name__ == "__main__":
    main()