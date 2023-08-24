FROM pytorch/pytorch

RUN apt update && apt install -y python3 python3-pip
RUN pip install setuptools_rust

RUN pip install setuptools wheel
RUN pip install datasets torch accelerate bitsandbytes transformers peft trl scipy

WORKDIR /training 

COPY . .

ENTRYPOINT [ "python", "llama2_python_data_trainer.py" ]

