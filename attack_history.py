"""
attack_history.py

This file contains the task of organizing and presenting
the attack history class.
"""
from dataclasses import dataclass
from typing import List
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
    attack_type: str
    raw_adversarial_examples: tf.data.Dataset
    clipped_adversarial_examples: tf.data.Dataset
    is_adversarial: tf.data.Dataset
    epsilons: List[float]
    model: tf.keras.Model
    attack_data: tf.data.Dataset
    attack_labels: tf.data.Dataset
    
    def get_robust_accuracy(self) -> float:
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
        plt.title("Perturbation of {} vs Accuracy of the Model".format(self.attack_type))
        plt.xlabel("epsilon")
        plt.ylabel("accuracy")
        plt.plot(self.epsilons, robust_accuracy.numpy())
        plt.savefig("output/epsilons_v_robust_acc_{}.png".format(self.attack_type))
        plt.clf()
        plt.close()

