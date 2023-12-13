from vllm import LLM
from vllm import SamplingParams
import gradio as gr
from huggingface_hub import snapshot_download
import os

class VLLMModel:
    """
    I basically took VLLMs quickstart guide + Modal's quickstart with VLLM guide and spliced them into 1 class
    Note: VLLM is really good for sending in multiple queries so user_questions is a list. Can change this

    To test this I used "mistralai/Mistral-7B-Instruct-v0.1"

    Let's see if we can make the max_tokens into an argument

    Pretty cool how just using LLM in VLLM downloads the tokenizer and the model at once (probably parallel)
    """
    
    def __init__(self, model):
        model_dir = f"/tmp/{model}/"
        if "meta" in model:
            try: 
                HUGGING_FACE_TOKEN = os.environ['HUGGING_FACE_TOKEN']
            except KeyError:
                raise Exception('Need to pass hugging face access token as environment variable HUGGING_FACE_TOKEN.')
        else:
            HUGGING_FACE_TOKEN = None

        snapshot_download(
            model,
            local_dir=model_dir,
            token=HUGGING_FACE_TOKEN
        )

        quantization = "AWQ" if "AWQ" in model else None
        max_model_len = 8192 if model == "TheBloke/Mistral-7B-Instruct-v0.1-AWQ" else None # Addresses CUDA OOM bug with specific model
        self.llm = LLM(model_dir, quantization=quantization, dtype='float16', max_model_len=max_model_len)
        self.template = """<s>[INST] <<SYS>>
                        You are a chatbot trained to answer a user's question succinctly without rephrasing. When answering, be direct and to the point. Don't include any [INST] or <<SYS>> tags in your response
                        <</SYS>>
                        {user} [/INST] """

    def generate(self, user_questions): 
        """
        user questions is a list of questions but can also be a single one

        
        """
        prompts = [
            self.template.format(user=q) for q in user_questions
        ]

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=500, # controls output length. leave others default
            presence_penalty=1.15,
        )
        
        result = self.llm.generate(prompts, sampling_params)
        
        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")

    def generate_gradio(self, message, history):
        """
        Hacky way to create the fn needed for a gradio UI
        """

        prompt = self.template.format(user=message)

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=500, # controls output length. leave others default
            presence_penalty=1.15,
        )

        result = self.llm.generate(prompt, sampling_params)

        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            tmp = output.outputs[0].text
            print(output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")

        return tmp

    def launch_chat(self):
        gr.ChatInterface(self.generate_gradio).queue().launch(server_name='0.0.0.0')