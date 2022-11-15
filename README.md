# MLAdversary

Develop various adversarial attacks against an image classifier to drastically reduce its performance, followed by improving its adversarial robustness. 

Team Name: CIA

Team Contact: Jocelyn Strmec, jocelyn.strmec@wsu.edu

Team Members: Abrhiram Bondada, Oscasavia Birugi, Noah Howell, Nathan Waltz (Development Lead Engineer), Jocelyn Strmec

![Team Information](https://user-images.githubusercontent.com/70173190/187054591-56b43cfc-ee6c-44de-922b-3a6eadb9b1ab.png)

## Project Steps

- [X] 1. Build a CNN and train it on an image dataset.

- [X] 2. Using Foolbox, create random noise and adversarial data to reduce accuracy of the CNN.

- [X] 3. Make model more robust.

## Startup Instructions


The most straightforward way to run the code in this repository is to install Docker and Visual Studio Code. After doing so, follow these steps:

1. Install the `Remote Development` extension made by Microsoft.

1. Open the folder containing the entirety of the repository in Visual Studio Code. 

1. Do `Ctrl+Shift+P`.

1. Select `Remote-Containers: Open Folder in Container`. 

After everything sets up, you should be able to get to work. To access the terminal to run `main.py`, you can do "Ctrl+Shift+\`". Then run `python main.py`!

There are a few command line arguments to be aware of...

| Argument Name  | Flag             | Args              | Description                                         |
| -------------- | ---------------- | ----------------- | --------------------------------------------------- |
| gpus           | --gpus           | None              | Whether to enable GPU accelerated machine learning. |
| load_model     | --load_model     | os.PathLike       | Loads a pretrained model from a path.               |
| model_output   | --model_output   | os.PathLike       | Outputs a model to a given path.                    |
| load_ensemble  | --load_ensemble  | List[os.PathLike] | A list of pretrained classifiers.                   |


## Project Resources

- https://github.com/bethgelab/foolbox

- https://foolbox.readthedocs.io/en/stable/modules/attacks.html 
