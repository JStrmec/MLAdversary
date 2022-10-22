FROM tensorflow/tensorflow:latest

# copy the files to the directory
COPY . /home/ml_adversary

WORKDIR /home/ml_adversary

# copy the dataset
RUN curl https://nathanwaltz.xyz/dataset/data.zip --output data.zip
RUN apt install unzip zip
RUN unzip data.zip -d .

# create the directories if they don't exist
RUN mkdir -p /home/ml_adversary/output
RUN mkdir -p /home/ml_adversary/saved_models

# install the requirements
RUN pip install -r requirements.txt

WORKDIR /home/ml_adversary
