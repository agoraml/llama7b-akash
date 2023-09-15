import torch
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from dataclasses import dataclass

@dataclass
class InferenceArguments():
    """
    Listing all the argument here that we will need
    A lot of them can come from the fine-tune script
    and im sure theres an easy way to pass those args in

    Additionally, we will need to update the req.txt
    with 
        pip install "fschat[model_worker,webui]"

    """
    # the following come from the finetuning script
    #model_name - ModelArguments
    #output_dir - TrainingArguments

    # 

def inference():
    # save the model
    model = AutoPeftModelForCausalLM.from_pretrained(
        "results/finalsave", #this should be output directory/finalsave or whatever we put in the finetune script
        device_map="auto", 
        torch_dtype=torch.bfloat16,
    )

    #merge and unload - keep for now
    merged_model = model.merge_and_unload(progressbar=True)

    # rest of the full save 
    output_merged_dir = "results/llama2/final_merged_model"
    os.makedirs(output_merged_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    merged_model.save_pretrained(output_merged_dir)
    tokenizer.save_pretrained(output_merged_dir)

    #after this we run the shell script (serve.sh) either through here or maybe a master
    # inference script. I think this could be called model_merge and the master script could
    # be called inference

if __name__ == "__main__":
    pass
