"""
model.py

This is where the actual task of binary classification
happens.
"""
import os
import random
from typing import Any, Optional, Tuple
from attack_history import AttackHistory
from config_loader import Config
import tensorflow as tf
import foolbox as fb
import eagerpy as ep


class ConvolutionalBlock(tf.keras.Model):
    """
    A convolutional block - used as a building block for the convolutional
    neural network that we are building.
    """

    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        pool_size: int,
        activation: tf.keras.activations,
    ) -> None:
        """
        Initializes a new instance of the ConvolutionalBlock.

        :param num_filters: The number of filters.
        :param kernel_size: The size of the kernel.
        :param pool_size: The pool size.
        :param activation: The activation function to use.
        """
        super(ConvolutionalBlock, self).__init__()
        self.conv_2d = tf.keras.layers.Conv2D(
            filters=num_filters, kernel_size=kernel_size
        )
        self.max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.activation = activation

    def call(self, inputs: Any, training: bool = False, mask: Any = None) -> None:
        """
        What happens when the class is "called" by Tensorflow.

        :param inputs: The inputs to the layer.
        :param training: Whether it is being used on training data.
        :param mask: The mask (I have no idea what it is, just implemented the interface).
        """
        x = self.conv_2d(inputs)
        x = self.max_pool_2d(x)
        x = self.batch_normalization(x, training=training)
        x = self.activation(x)
        return x


class Model:
    def __init__(self, config: Config, model_path: Optional[os.PathLike] = None):
        """
        Initializes a new instance of the model class.

        We are using a similar architecture as found here, with a few differences:
        https://keras.io/examples/vision/3D_image_classification/

        No way could I pull this off without some sort of reference.

        :param config: The configuration parameters.
        :param model_path: The absolute path to a saved model.
        """
        self.config = config
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    (
                        self.config.dataset_config.image_width,
                        self.config.dataset_config.image_height,
                        3,
                    )
                ),
                ConvolutionalBlock(32, 3, 2, tf.keras.activations.relu),
                ConvolutionalBlock(32, 3, 2, tf.keras.activations.relu),
                ConvolutionalBlock(64, 3, 2, tf.keras.activations.relu),
                ConvolutionalBlock(64, 3, 2, tf.keras.activations.relu),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax),
            ]
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.model_config.learning_rate
            ),
            metrics=["accuracy"],
        )

        if model_path:
            self._load_model(model_path)

    def fit_model(
        self,
        training_data: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        save_directory: Optional[os.PathLike] = None,
    ) -> tf.keras.callbacks.History:
        """
        Fits the model on training data. We are using an early stopping callback
        to stop training if necessary.

        :param training_data: The training data that is being used.
        :param validation_data: The validation data that is being used.
        :param save_directory: The directory to save the model at.
        """
        checkpoint = None
        if save_directory is not None:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_directory, save_best_only=True
            )

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
        if checkpoint is not None:
            callbacks += [checkpoint]

        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.config.model_config.num_epochs,
            callbacks=callbacks,
            batch_size=self.config.model_config.batch_size,
            validation_batch_size=self.config.model_config.batch_size,
        )
        return history

    def make_predictions(self, data) -> None:
        """
        Makes predictions on given data.

        :param data: The data to make predictions on.
        """

        return self.model.predict(data)

    def _load_model(self, path: os.PathLike) -> None:
        """
        Loads a model from an absolute path.

        :param path: The path to load the model from.
        """
        if not os.path.exists(path):
            return

        self.model.load_weights(path)

    @staticmethod
    def _random_batch_from_tf_dataset(
        data: tf.data.Dataset, seed: int = 0
    ) -> Tuple[ep.types.NativeTensor, ep.types.NativeTensor]:
        """
        Takes a random batch from a tf_dataset.

        :param data: The Tensorflow dataset that is being used.
        :param seed: The random seed.

        :return: A tuple containing a batch of images and labels.
        """
        random.seed(seed)
        index = random.randint(0, len(data) - 1)
        data = data.skip(index)
        image, label = next(data.as_numpy_iterator())
        image = ep.astensor(tf.convert_to_tensor(image))
        label = ep.astensor(
            tf.reshape(tf.convert_to_tensor(label, dtype=tf.int64), [-1])
        )
        return image, label

    def linf_projected_gradient_descent_attack(self, foolbox_model: fb.TensorFlowModel, images: ep.types.NativeTensor, labels: ep.types.NativeTensor, epsilons: list[float]) -> AttackHistory:
        """
        Performs a Linf Projected Gradient Descent Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.LinfPGD()
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory("Linf Projected Gradient Descent Attack",raw_advs, clipped_advs, success, epsilons, foolbox_model, images, labels)

    def fast_gradient_descent_attack(self, foolbox_model: fb.TensorFlowModel, images: ep.types.NativeTensor, labels: ep.types.NativeTensor, epsilons: list[float]) -> AttackHistory:
        """
        Performs a Fast Gradient Method (FGM) Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.FGM()
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory("Fast Gradient Method (FGM) Attack",raw_advs, clipped_advs, success, epsilons, foolbox_model, images, labels)

    def deepfool_attack(self, foolbox_model: fb.TensorFlowModel, images: ep.types.NativeTensor, labels: ep.types.NativeTensor, epsilons: list[float]) -> AttackHistory:
        """
        Performs a DeepFool Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.DeepFoolAttack()
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory("DeepFool Attack",raw_advs, clipped_advs, success, epsilons, foolbox_model, images, labels)

    def linf_addative_noise_attack(self, foolbox_model: fb.TensorFlowModel, images: ep.types.NativeTensor, labels: ep.types.NativeTensor, epsilons: list[float])-> AttackHistory:
        """
        Performs a Linf Addative Noise Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.LinfAdditiveUniformNoiseAttack()
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory("Linf Addative Noise Attack",raw_advs, clipped_advs, success, epsilons, foolbox_model, images, labels)

    def preform_attacks(self, data: tf.data.Dataset, epsilons: list[float]) -> list[AttackHistory]:
        """
        Performs all attacks on the model.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # create a foolbox model
        model = fb.TensorFlowModel(self.model, bounds=(0, 255))
        # get a random batch from the data
        image, label = self._random_batch_from_tf_dataset(data)
        # preform the attacks
        linf_pgd_attack_history = self.linf_projected_gradient_descent_attack(model, image, label, epsilons)
        deepfool_attack_history = self.deepfool_attack(model, image, label, epsilons)
        fgm_attack_history = self.fast_gradient_descent_attack(model, image, label, epsilons)
        linf_addative_noise_attack_history = self.linf_addative_noise_attack(model, image, label, epsilons)
        return [linf_pgd_attack_history, deepfool_attack_history, fgm_attack_history, linf_addative_noise_attack_history]
