"""
model.py

This is where the actual task of binary classification
happens.
"""
import os
import random
from typing import Any, Optional, Tuple, List, Callable, Iterable
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


class ConvolutionalModel(tf.keras.Model):
    """
    The convolutional model.
    """

    def __init__(
        self,
        filter_multiplier: int,
        num_sibling_blocks: int,
        dense_output_size: int,
        dropout_ratio: float,
    ):
        """
        Initializes a new instance of the ConvolutionalModel class.

        We are using a similar architecture as found here, with a few differences:
        https://keras.io/examples/vision/3D_image_classification/

        :param filter_multiplier: The multiplier for the number of filters.
        :param num_sibling_blocks: The number of sibling blocks.
        :param dense_output_size: The size of the intermediate dense output size.
        :param dropout_ratio: The ratio of neurons to "dropout".
        """
        super(ConvolutionalModel, self).__init__()
        self.convolutional_portion = tf.keras.Sequential(
            layers=[
                ConvolutionalBlock(filter_multiplier, 3, 2, tf.keras.activations.relu)
                for _ in range(num_sibling_blocks)
            ]
            + [
                ConvolutionalBlock(
                    2 * filter_multiplier, 3, 2, tf.keras.activations.relu
                )
                for _ in range(num_sibling_blocks)
            ]
        )
        self.average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_portion = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(
                    units=dense_output_size, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dropout(dropout_ratio),
                tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax),
            ]
        )

    def call(self, inputs: Any, training: bool = False, mask: Any = None) -> None:
        """
        What happens when the class is "called" by Tensorflow.

        :param inputs: The inputs to the layer.
        :param training: Whether it is being used on training data.
        :param mask: The mask (I have no idea what it is, just implemented the interface).
        """
        x = self.convolutional_portion(inputs)
        x = self.average_pooling(x)
        x = self.dense_portion(x)
        return x


class Model:
    def __init__(self, config: Config, model_path: Optional[os.PathLike] = None):
        """
        Initializes a new instance of the model class.

        :param config: The configuration parameters.
        :param model_path: The absolute path to a saved model.
        """
        self.config = config
        models = [
            ConvolutionalModel(32, 2, 1024, 0.3)
            for _ in range(config.model_config.ensemble_size)
        ]
        model_input = tf.keras.layers.Input(
            shape=(
                self.config.dataset_config.image_width,
                self.config.dataset_config.image_height,
                self.config.dataset_config.num_channels,
            )
        )
        model_outputs = [model(model_input) for model in models]
        output = tf.keras.layers.Average()(model_outputs)
        self.model = tf.keras.Model(inputs=model_input, outputs=output)
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

        self.model.save("saved_model/model")
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

        self.model = tf.keras.models.load_model(path)

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

    def _linf_projected_gradient_descent_attack(
        self,
        foolbox_model: fb.TensorFlowModel,
        images: ep.types.NativeTensor,
        labels: ep.types.NativeTensor,
        epsilons: List[float],
    ) -> AttackHistory:
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

        return AttackHistory(
            "Linf Projected Gradient Descent Attack",
            raw_advs,
            clipped_advs,
            success,
            epsilons,
            foolbox_model,
            images,
            labels,
        )

    def _fast_gradient_descent_attack(
        self,
        foolbox_model: fb.TensorFlowModel,
        images: ep.types.NativeTensor,
        labels: ep.types.NativeTensor,
        epsilons: List[float],
    ) -> AttackHistory:
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

        return AttackHistory(
            "Fast Gradient Method Attack",
            raw_advs,
            clipped_advs,
            success,
            epsilons,
            foolbox_model,
            images,
            labels,
        )

    def _deepfool_attack(
        self,
        foolbox_model: fb.TensorFlowModel,
        images: ep.types.NativeTensor,
        labels: ep.types.NativeTensor,
        epsilons: List[float],
    ) -> AttackHistory:
        """
        Performs a Linf Deep Fool Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.LinfDeepFoolAttack()
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory(
            "Linf Deep Fool Attack",
            raw_advs,
            clipped_advs,
            success,
            epsilons,
            foolbox_model,
            images,
            labels,
        )

    def _linf_iterative_attack(
        self,
        foolbox_model: fb.TensorFlowModel,
        images: ep.types.NativeTensor,
        labels: ep.types.NativeTensor,
        epsilons: List[float],
    ) -> AttackHistory:
        """
        Performs a Linf Basic Iterative Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.LinfBasicIterativeAttack()
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory(
            "Linf Basic Iterative Attack",
            raw_advs,
            clipped_advs,
            success,
            epsilons,
            foolbox_model,
            images,
            labels,
        )

    def _inversion_attack(
        self,
        foolbox_model: fb.TensorFlowModel,
        images: ep.types.NativeTensor,
        labels: ep.types.NativeTensor,
        epsilons: List[float],
    ) -> AttackHistory:
        """
        Performs a Inversion Attack.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :return: The attack history.
        """
        # define the attack
        attack = fb.attacks.InversionAttack(distance=fb.distances.LpDistance(p=2))
        raw_advs, clipped_advs, success = attack(
            foolbox_model, images, labels, epsilons=epsilons
        )

        return AttackHistory(
            "Inversion Attack",
            raw_advs,
            clipped_advs,
            success,
            epsilons,
            foolbox_model,
            images,
            labels,
        )

    def preform_attacks(
        self,
        data: tf.data.Dataset,
        epsilons: List[float],
        attack_methods: Iterable[
            Callable[
                [
                    fb.TensorFlowModel,
                    ep.types.NativeTensor,
                    ep.types.NativeTensor,
                    List[float],
                ],
                AttackHistory,
            ]
        ],
    ) -> List[AttackHistory]:
        """
        Perform attacks on the model.

        :param model: The model to attack.
        :param images: The images to attack.
        :param labels: The labels of the images.
        :param epsilons: The epsilons to use.
        :param attack_methods: The attacks to perform.
        :return: The attack history.
        """
        model = fb.TensorFlowModel(self.model, bounds=(0, 255))
        image, label = self._random_batch_from_tf_dataset(data)
        return [method(model, image, label, epsilons) for method in attack_methods]
