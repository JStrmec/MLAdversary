"""
main.py

Where the entry point of the application resides.
"""
import sys
from data_loader import DataLoader
from config_loader import ConfigLoader
from model import Model

import matplotlib.pyplot as plt

project_config_loader = ConfigLoader()
project_config = project_config_loader.get_config()

project_data_loader = DataLoader(project_config)
project_data = project_data_loader.load_data()

# create and train the model
model = Model(project_config)

# Jocelyn - You will need to either load a model from the saved weights or train a model
history = model.fit_model(project_data.train, project_data.validation, "saved_models/model")
#history = None

# this is just testing to make sure we can get foolbox to work, lets provide some
# analysis about the attack's effectiveness - examples in the book
attack_history = model.linf_projected_gradient_descent_attack(project_data.train)

# Attack Analysis
# print attack history
attack_history.print()

if not history:
    sys.exit(1)

# plotting code sourced from https://keras.io/examples/vision/3D_image_classification/
# because it has good examples of taking metrics
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

plt.savefig("output/results.png")
