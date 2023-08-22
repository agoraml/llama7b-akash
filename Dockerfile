FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="Ishan Dhanani <ishan@agoralabs.io"
LABEL version="0.0.1"
LABEL description="Training Llama2 on python code data"

WORKDIR /training 

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "llama2_python_data_trainer.py" ]

