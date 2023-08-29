FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt update && apt install -y python3 python3-pip
RUN pip install setuptools_rust setuptools wheel
RUN pip install peft accelerate datasets bitsandbytes nvidia-ml-py3 scipy trl torch
RUN pip install --upgrade git+https://github.com/huggingface/transformers

WORKDIR /training 

COPY . .

ENTRYPOINT [ "python", "agora_python_bot.py" ]

