# MLAdversary

Develop various adversarial attacks against an image classifier to drastically reduce its performance, followed by improving its adversarial robustness. 

Team Name: CIA

Team Contact: Jocelyn Strmec, jocelyn.strmec@wsu.edu

Team Members: Abrhiram Bondada, Oscasavia Birugi, Noah Howell, Nathan Waltz (Development Lead Engineer), Jocelyn Strmec

![Team Information](https://user-images.githubusercontent.com/70173190/187054591-56b43cfc-ee6c-44de-922b-3a6eadb9b1ab.png)

## Project Steps

- [X] 1. Build a CNN and train it on an image dataset.

- [ ] 2. Using Foolbox, create random noise and adversarial data to reduce accuracy of the CNN.

- [ ] 3. Train on adversarial data to increase accuracy of the CNN model.

## Startup Instructions

In order to run this repository, you will want to first install some version of Python 3 (it doesn't matter since you will be running this in a Docker container anyways, just used for the startup script) and Docker, then follow the instructions to install Tensorflow's Docker container [here](https://www.tensorflow.org/install/docker). This is by far the easiest way to get Tensorflow all set up, and some convenience scripts were written to streamline this setup process for you. 

Then, you will want to run the environment setup script!

`$ python env_setup.py`

Afterwards, you will want to build the Docker container as follows:

`$ sudo docker build -t ml_adversary .`

Then, you can access the environment as follows:

```
$ sudo docker run \ 
       --gpus all \
       --rm \ 
       -it \
       --name ml_adversary_container \
       -v "$(pwd)"/output,target=/home/ml_adversary/output \
       -v "$(pwd)"/saved_models,target=/home/ml_adversary/saved_models \
       ml_adversary bash
```

## Project Resources

- https://github.com/bethgelab/foolbox

- https://foolbox.readthedocs.io/en/stable/modules/attacks.html 
