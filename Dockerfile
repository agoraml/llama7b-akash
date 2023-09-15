FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /training 

COPY . .

RUN chmod 755 src/inference/serve.sh
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "src/finetune_inference_flow.py"]