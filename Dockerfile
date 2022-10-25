FROM tensorflow/tensorflow:latest

WORKDIR /home/ml_adversary

COPY requirements.txt .

# perform administravia
RUN apt-get update \
    && apt-get install git -y \
    && pip install -r requirements.txt


