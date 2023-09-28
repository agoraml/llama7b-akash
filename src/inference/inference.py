import torch
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
import gradio as gr
from src.training.finetune import print_gpu_utilization

class InferenceUI:
    def __init__(self, model_name: str, output_dir: str):
        model = AutoPeftModelForCausalLM.from_pretrained(
            os.path.join(output_dir, "checkpoint-final"), #this should be output directory/finalsave or whatever we put in the finetune script
            device_map="auto", 
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_gpu_utilization()

        llama_pipeline = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer = tokenizer 
        )
        
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = llama_pipeline
        
    def format_message(self, message: str, history: list, memory_limit: int = 3) -> str:
        """
        Formats the message based on our input. Does not handle history due to a short context and the 
        prompt being so verbose

        Parameters:
            message (str): Current message to send.
            history (list): Past conversation history.
            memory_limit (int): Limit on how many past interactions to consider.

        Returns:
            str: Formatted message string
        """
        instruction=f"Create a code snippet written in Python. {message}"
        input = "" #empty string for now because user will just be passing in instruction and not an output for our purposes
        prompt = f"""### Instruction:
                    Use the Task below and the Input given to write the Response, which is Python code that can solve the Task.
                    
                    ### Task:
                    {instruction}
                    
                    ### Input:
                    {input}
                    
                    ### Response:
                    """

        return prompt
    
    # Generate a response from the Llama model
    def get_llama_response(self, message: str, history: list) -> str:
        """
        Generates a conversational response from the Llama model.

        Parameters:
            message (str): User's input message.
            history (list): Past conversation history.

        Returns:
            str: Generated response from the Llama model.
        """
        query = self.format_message(message, history)
        response = ""

        sequences = self.pipeline(
            query,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1024,
        )

        generated_text = sequences[0]['generated_text']
        response = generated_text[len(query):]  # Remove the prompt from the output

        print("Chatbot:", response.strip())
        return response.strip()
    
    def launch_chat(self):
        gr.ChatInterface(self.get_llama_response).queue().launch(server_name='0.0.0.0') 

def launch_inference(model_name : str, output_dir : str):
    inference_ui = InferenceUI(model_name, output_dir)
    inference_ui.launch_chat()