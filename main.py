"""
main.py

Where the entry point of the application resides.
"""
import random
from typing import List, Optional, Tuple
import tensorflow as tf
import argparse
from attack_history import AttackHistory
from data_loader import DataLoader, ModelData, random_noise_transformation
from config_loader import Config, ConfigLoader
from model import Model

import matplotlib.pyplot as plt

# parse CLI arguments
parser = argparse.ArgumentParser(
    prog="ML Adversary", description="Adversarial Machine Learning."
)
parser.add_argument(
    "--gpus",
    help="Whether to use GPU accelerated ML.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--load_model", help="Load a model from a path.", type=str, default=None
)
parser.add_argument(
    "--model_output",
    help="The absolute path to save the model to.",
    type=str,
    default="saved_model/model",
)
parser.add_argument(
    "--load_ensemble",
    help="Loads an ensemble from memory.",
    type=str,
    nargs="+"
)
args = parser.parse_args()


def main() -> None:
    """
    Script entry point.
    """
    # enable/disable GPU access
    set_gpu_access(parsed_args=args)

    # load the project configuration
    project_config_loader = ConfigLoader()
    project_config = project_config_loader.get_config()

    # load the project data
    project_data_loader = DataLoader(project_config)
    project_data = project_data_loader.load_data(random_noise_transformation)

    # set the random seed
    random.seed(project_config.model_config.seed)
    tf.random.set_seed(project_config.model_config.seed)

    # obtain model and training history
    history, model = obtain_model_for_robustness_testing(
        args, project_data, project_config
    )
    display_history_metrics(history)

    # attack the model
    attack_history = perform_attacks(model, project_data, project_config)
    analyze_attack_history(attack_history)


def obtain_model_for_robustness_testing(
    parsed_args: argparse.Namespace, data: ModelData, config: Config
) -> Tuple[Optional[tf.keras.callbacks.History], Model]:
    """
    Obtains a model for robustness testing.

    :parsed_args: The argument namespace.
    :param data: The data to use.
    :param config: The model configuration file.
    :return: The model training history, and the model.
    """
    history = None
    model = Model(config)
    if parsed_args.load_model is not None:
        print("Loading a singular model from memory.")
        model = Model(config, parsed_args.load_model)
    elif parsed_args.load_ensemble:
        print("Loading an ensemble from memory!")
        model.load_ensemble_models(parsed_args.load_ensemble)
        history = model.fit_model(data.train, data.validation, parsed_args.model_output)
    else:
        print("Training a new model")
        history = model.fit_model(data.train, data.validation, parsed_args.model_output)
    return history, model


def set_gpu_access(parsed_args: argparse.Namespace) -> None:
    """
    Determines whether there is GPU access from the args namespace.

    :parsed_args: The argument namespace.
    """
    if not parsed_args.gpus:
        tf.config.set_visible_devices([], "GPU")


def display_history_metrics(history: Optional[tf.keras.callbacks.History]) -> None:
    """
    Displays relevant accuracy/loss metrics for a given keras
    callback history.

    :param history: The model history.
    """
    if history is None:
        return

    # plotting code sourced from https://keras.io/examples/vision/3D_image_classification/
    # because it has good examples of taking metrics
    _, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["accuracy", "loss"]):
        ax[i].plot(history.history[metric])
        ax[i].plot(history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

    plt.savefig("output/results.png")
    plt.close()


def perform_attacks(
    model: Model, data: ModelData, config: Config
) -> List[AttackHistory]:
    """
    Perform attacks on the model.

    :param model: The model to perform attacks on.
    :param data: The data to use.
    :param config: The model configuration file.
    :return: A list of relevant attack histories.
    """
    return model.preform_attacks(
        data.train,
        config.attack_config.epsilons,
        [
            model._inversion_attack,
            model._deepfool_attack,
            model._linf_iterative_attack,
            model._linf_projected_gradient_descent_attack,
            model._fast_gradient_descent_attack,
        ],
    )


def analyze_attack_history(attack_history: List[AttackHistory]) -> None:
    """
    Perform analysis of attack history.

    :param attack_history: The attack history.
    """
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
    plt.close()

if __name__ == "__main__":
    main()
