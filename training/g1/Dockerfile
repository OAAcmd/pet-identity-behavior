# G1 training container (GPU). Build from the repository root:
#   docker build -f training/g1/Dockerfile -t petperception-g1 .
# The base image ships CPython 3.11 and TensorFlow 2.15 with CUDA 12.2 / cuDNN 8.9.
FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /work
COPY training/g1/requirements.lock.txt /tmp/requirements.lock.txt
# tensorflow itself comes from the base image; everything else is pinned by the lock file.
RUN grep -v '^tensorflow==' /tmp/requirements.lock.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt
COPY training/g1/ /work/training/g1/
COPY results/g1/classes.json /work/results/g1/classes.json
ENTRYPOINT ["python", "/work/training/g1/train.py"]
