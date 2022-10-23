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


# epsilons
epsilons = [0.0,0.0002,0.0005,0.0008,0.001,0.0015,0.002,0.003,0.01,0.1,0.3,0.5,1.0]

# preforms 4 attacks on the model and returns the history of each attack in a list as 
# [LPDG, DF, FGM, LAN]
attack_history = model.preform_attacks(project_data.train, epsilons)

# Attack Analysis
for attack in attack_history:
    attack.analysis()
    # Plot single comparison of adversarial anaylsis
    plt.plot(attack.epsilons, attack.get_robust_accuracy(), "*-", label = attack.attack_type)
plt.legend(loc="upper left")
plt.set_xlabel("epsilon")
plt.set_ylabel("accruacy")
plt.savefig("output/preturbation_comparion.png")
plt.clf()

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
