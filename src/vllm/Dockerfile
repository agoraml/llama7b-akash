FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app 

COPY . .

RUN ["pip", "install", "-r", "requirements.txt"]

ENTRYPOINT ["python3", "vllm_inference.py"]