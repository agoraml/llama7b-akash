#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

model_path="$1"

echo 'Starting controller...'
python -u -m fastchat.serve.controller --host 0.0.0.0 > ~/controller.log 2>&1 &
sleep 10
echo 'Starting model worker...'
python -u -m fastchat.serve.model_worker \
        --model-path "$model_path" --host 0.0.0.0 2>&1 \
        | tee model_worker.log &

echo 'Waiting for model worker to start...'
while ! `cat model_worker.log | grep -q 'Uvicorn running on'`; do sleep 1; done

echo 'Starting gradio server...'
python -u -m fastchat.serve.gradio_web_server --share | tee ~/gradio.log