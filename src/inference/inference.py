import torch
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def save_model_for_inference(model_name : str, new_model_name : str, output_dir : str):
    # save the model
    model = AutoPeftModelForCausalLM.from_pretrained(
        os.path.join(output_dir, "checkpoint-final"), #this should be output directory/finalsave or whatever we put in the finetune script
        device_map="auto", 
        torch_dtype=torch.bfloat16,
    )

    #merge and unload - keep for now
    merged_model = model.merge_and_unload(progressbar=True)

    output_merged_dir = os.path.join(output_dir, new_model_name)
    os.makedirs(output_merged_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    merged_model.save_pretrained(output_merged_dir)
    tokenizer.save_pretrained(output_merged_dir)

    #after this we run the shell script (serve.sh) either through here or maybe a master
    # inference script. I think this could be called model_merge and the master script could
    # be called inference