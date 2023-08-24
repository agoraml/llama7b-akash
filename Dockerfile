FROM python:3.9-slim
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# LABEL maintainer="Ishan Dhanani <ishan@agoralabs.io"
# LABEL version="0.0.1"
# LABEL description="Training Llama2 on python code data"

# WORKDIR /training 

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# ENTRYPOINT [ "python", "llama2_python_data_trainer.py" ]

# FROM ubuntu:20.04

# # Installing lambda stack repositories
# RUN LAMBDA_REPO=$(mktemp) && \
#     wget -O${LAMBDA_REPO} https://lambdalabs.com/static/misc/lambda-stack-repo.deb && \
#     sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} && \
#     sudo apt-get update && sudo apt-get install -y lambda-stack-cuda \
#     sudo reboot

# # Installing docker & nvidia-container-toolkit
# RUN sudo apt-get install -y docker.io nvidia-container-toolkit

