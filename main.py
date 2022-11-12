"""
main.py

Where the entry point of the application resides.
"""
import sys
import tensorflow as tf
from data_loader import DataLoader, random_noise_transformation
from config_loader import ConfigLoader
from model import Model

import matplotlib.pyplot as plt

# Hide GPU from visible devices (used for robustness analysis, not)
tf.config.set_visible_devices([], "GPU")

project_config_loader = ConfigLoader()
project_config = project_config_loader.get_config()

project_data_loader = DataLoader(project_config)
project_data = project_data_loader.load_data(random_noise_transformation)

# create and train the model
model = Model(project_config, "saved_model/model")

# history = model.fit_model(
#     project_data.train, project_data.validation, "saved_models/model"
# )

# epsilons for attacks
epsilons = [
    0.0,
    0.00025,
    0.0005,
    0.0008,
    0.001,
    0.0015,
    0.0025,
    0.005,
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    0.8,
    1.0,
]

# perform inversion and deepfool attacks
attack_history = model.preform_attacks(
    project_data.train,
    epsilons,
    [
        model._inversion_attack,
        model._deepfool_attack,
        model._linf_iterative_attack,
        model._linf_projected_gradient_descent_attack,
        model._fast_gradient_descent_attack,
    ],
)

# ascertain analysis
for attack in attack_history:
    attack.analysis()

# plot comparisons
for attack in attack_history:
    plt.plot(
        attack.epsilons,
        (attack.get_robust_accuracy()).numpy(),
        label=attack.attack_type,
    )

plt.title("Perturbation of Attacks vs Accuracy of the Model")
plt.legend(loc="upper right")
plt.xlabel("epsilon")
plt.ylabel("accuracy")
plt.savefig("output/preturbation_comparsion.png")
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
