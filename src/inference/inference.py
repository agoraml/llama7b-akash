import torch
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from src.training.finetune import print_gpu_utilization
import gradio as gr


class AgoraInference:
    def __init__(self, model_name, output_dir):
        model = AutoPeftModelForCausalLM.from_pretrained(
            os.path.join(
                output_dir, "checkpoint-final"
            ),  # this should be output directory/finalsave or whatever we put in the finetune script
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
            tokenizer=tokenizer,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = llama_pipeline

    def format_message(self, message: str, history: list, memory_limit: int = 3) -> str:
        """
        Formats the message and history for the Llama model.

        Parameters:
            message (str): Current message to send.
            history (list): Past conversation history.
            memory_limit (int): Limit on how many past interactions to consider.

        Returns:
            str: Formatted message string
        """

        SYSTEM_PROMPT = """<s>[INST] <<SYS>>
                You are a helpful chatbot that has been optimized for coding in Python. Limit all 
                prose in your responses. Provide correct Python code without making up any functions at all. 
                Make sure to add comments
                <</SYS>>"""

        # always keep len(history) <= memory_limit
        if len(history) > memory_limit:
            history = history[-memory_limit:]

        if len(history) == 0:
            return SYSTEM_PROMPT + f"{message} [/INST]"

        formatted_message = (
            SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"
        )

        # Handle conversation history
        for user_msg, model_answer in history[1:]:
            formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

        # Handle the current message
        formatted_message += f"<s>[INST] {message} [/INST]"

        return formatted_message

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

        generated_text = sequences[0]["generated_text"]
        response = generated_text[len(query) :]  # Remove the prompt from the output

        print("Chatbot:", response.strip())
        return response.strip()

    def launch_chat(self):
        gr.ChatInterface(self.get_llama_response).launch(share=True)
