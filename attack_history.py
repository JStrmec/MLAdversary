"""
attack_history.py

This file contains the task of organizing and presenting
the attack history class.
"""
from dataclasses import dataclass
import tensorflow as tf
import matplotlib.pyplot as plt

@dataclass
class AttackHistory:
    """
    Class for keeping track of the history of attacks inculding
    raw adversarial examples, clipped adversarial examples, and
    third tensor contains a boolean for each sample, indicating 
    which samples are true adversarials that are both 
    misclassified and within the epsilon balls around the clean
    samples.
    """

    raw_adversarial_examples: tf.data.Dataset
    clipped_adversarial_examples: tf.data.Dataset
    is_adversarial: tf.data.Dataset
    epsilons: list[float]
    model: tf.keras.Model
    attack_data: tf.data.Dataset
    attack_labels: tf.data.Dataset

    def __init__(
        self,
        raw_adversarial_examples: tf.data.Dataset,
        clipped_adversarial_examples: tf.data.Dataset,
        is_adversarial: tf.data.Dataset,
        epsilons: list[float],
        model: tf.keras.Model,
        attack_data: tf.data.Dataset,
        attack_labels: tf.data.Dataset,
    ) -> None:
        """
        Initializes a new instance of the AttackHistory class.

        :param raw_adversarial_examples: The raw adversarial examples.
        :param clipped_adversarial_examples: The clipped adversarial examples.
        :param is_adversarial: The boolean indicating whether the sample is an adversarial example.
        :param epsilons: The epsilons used in the attack.
        :param model: The model used in the attack.
        :param attack_data: The data used in the attack.
        :param attack_labels: The labels used in the attack.
        """
        self.raw_adversarial_examples = raw_adversarial_examples
        self.clipped_adversarial_examples = clipped_adversarial_examples
        self.is_adversarial = is_adversarial
        self.epsilons = epsilons
        self.model = model
        self.attack_data = attack_data
        self.attack_labels = attack_labels

    
    def get_robust_accuracy(self)-> float:
        """
        Returns the robust accuracy of the attack history.

        :return: The robust accuracy of the attack history.
        """
        return 1 - self.is_adversarial.float32().mean(axis=-1)

    def analysis(self) -> None:
        """
        Prints the history of the attack.
        """
        # plot the robust accuracy
        print("robust accuracy for perturbations with")
        robust_accuracy = self.get_robust_accuracy()
        for eps, acc in zip(self.epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

        plt.plot(self.epsilons, robust_accuracy.numpy())
        plt.savefig("output/eplisons_v_robust_acc.png")