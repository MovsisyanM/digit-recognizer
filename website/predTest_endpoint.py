import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.preprocessing import minmax_scale
from flask import Flask, Response, request
import tensorflow as tf
from keras import backend as K
import matplotlib.patches as patches
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.layers.experimental.preprocessing import RandomTranslation, RandomRotation, RandomWidth, RandomHeight, RandomZoom, Resizing
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, BatchNormalization, Reshape, GlobalAveragePooling2D, LeakyReLU, Layer, Concatenate, Input


app = Flask(__name__)

# region custom stuff


class GateOfLearning(Callback):
    """Increases learning rate when stuck at extrema, a friend to ReduceLROnPlateau, ModelCheckpoint callbacks.\n

    \n
    Args:\n
        `monitor`: quantity to be monitored.\n
        `factor`: factor by which the learning rate will be reduced. Must be multitudes greater than that of the ReduceLROnPlateau\n
            `new_lr = lr * factor`.\n
        `patience`: number of epochs with no improvement after which learning rate will be increased. Must be greater \
            than that of the ReduceLROnPlateau `(6 by default)`\n
        `verbose`: int. 0: quiet, 1: update messages. `(1 by default)`\n
        `mode`: one of `{'min', 'max'}`. In `'min'` mode, the learning rate will be increased when the quantity\
             monitored has stopped decreasing; in `'max'` mode it will be increased when the quantity monitored has stopped increasing.\n
        `cooldown`: number of epochs to wait before resuming normal operation afterlr has been reduced. `(0 by default)`\n
        `max_lr`: upper bound on the learning rate. `(initial value * 50 by default)`\n
    """

    def __init__(self, monitor="val_loss", factor=15.0, patience=6, verbose=1, mode="min", cooldown=0, max_lr=999):
        # Sanity check
        if factor <= 1.0:
            raise ValueError(
                "GateOfLearning does not support a factor <= 1.0.")

        if mode not in ["min", "max"]:
            raise ValueError(
                f"GateOfLearning does not support a mode '{mode}'. Use 'min' or 'max' instead.")

        # Init
        super(GateOfLearning, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.objective = min if mode == "min" else max
        self.cooldown = cooldown
        self.max_lr = max_lr

        self.backup = (monitor, factor, patience,
                       verbose, mode, cooldown, max_lr)

        self.observations = []
        self.lr_history = []
        self.last_opened = 0

    def _reset(self):
        """Reset state"""
        self.monitor, self.factor, self.patience, self.verbose, self.mode, self.cooldown, self.max_lr = self.backup

        self.observations = []
        self.lr_history = []
        self.last_opened = 0

    def on_train_begin(self, logs=None):
        """Training start handler"""
        self._reset()

    def open_gate(self):
        """Increases learning rate"""
        new_lr = self.lr_history[-1] * self.factor

        assert new_lr > self.lr_history[-1], f"old: {self.lr_history[-1]}, new: {new_lr}"

        if new_lr > self.max_lr:

            if self.verbose:
                print("Learning rate diverged. You can solve this problem by using a faster ReduceLROnPlateau, \
                    a smaller factor, or a bigger patience/cooldown. Make sure the objective is appropriate.")
        else:
            old_lr = float(self.model.optimizer.learning_rate)
            self.model.optimizer.learning_rate = new_lr
            if self.verbose:
                print(
                    f"\nGateOfLearning: Learning rate increased from {old_lr} to {float(self.model.optimizer.learning_rate)}")

    def on_epoch_end(self, epoch, logs=None):
        """Epoch end handler"""
        # Log learning rate.
        self.lr_history.append(logs["lr"])

        # Set the maximum learning rate to the initial or otherwise specified maximum learning rate
        if len(self.lr_history) <= 1:
            self.max_lr = min(self.max_lr, 50 * self.lr_history[0])

        # Check if the metric is reported, otherwise use default metrics.
        if self.monitor not in logs.keys():
            initMetric = self.monitor
            self.monitor = "val_loss" if "val_loss" in logs.keys() else "loss"
            if self.verbose:
                print(
                    f"\nGateOfLearning: The '{initMetric}' metric was never reported. Using '{self.monitor}' instead.\n")

        # Log metric
        self.observations.append(logs[self.monitor])

        # Check if it is too early for an opening
        if len(self.observations) <= self.patience:
            return

        # Check if there is no improvement
        if self.objective(self.observations[-self.patience:]) == self.observations[-self.patience]:
            if epoch - self.last_opened > self.cooldown:
                self.open_gate()
                self.last_opened = epoch
                self.observations = [self.observations[-self.patience]]


class SingularityExtractor2D(Layer):
    """
    Accentuates pixels that differ from its surrounding pixels. 
    """

    def __init__(self, degree=6, kernel_size=3, padding="SYMMETRIC", margin=1, **kwargs):
        if not ((kernel_size % 2) and kernel_size >= 3 and isinstance(kernel_size, int)):
            raise ValueError("kernel_size: value must be odd, >= 3")

        if not (margin >= 1 and isinstance(margin, int)):
            raise ValueError("margin: must be integer >= 1")

        if padding not in ["CONSTANT", "SYMMETRIC", "REFLECT"]:
            raise ValueError(
                "padding: must be one of ['CONSTANT', 'SYMMETRIC', 'REFLECT']")

        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.margin = margin

        self.radius = int((kernel_size - 1)/2)

        if "trainable" in kwargs.keys():
            super(SingularityExtractor2D, self).__init__(**kwargs)
        else:
            super(SingularityExtractor2D, self).__init__(
                trainable=False, **kwargs)

    # region Keras api

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("input must be 4D (batch_size, x, y, channels)")

        super(SingularityExtractor2D, self).build(input_shape)

        # Lazy load conv kernel
        self.ones = tf.ones(
            (self.kernel_size, self.kernel_size, input_shape[-1], input_shape[-1]))

    def get_config(self):
        """Layer to dict = serializability"""

        config = super(SingularityExtractor2D, self).get_config()
        config["degree"] = self.degree
        config["kernel_size"] = self.kernel_size
        config["padding"] = self.padding
        config["margin"] = self.margin
        config["dtype"] = self.dtype

        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            config["degree"],
            config["kernel_size"],
            config["padding"],
            config["margin"],
            dtype=config["dtype"]
        )

    def call(self, input_data):
        return self.extract(input_data, "keras")

    # endregion Keras api

    # region Scikit api

    def fit(self, x, *args, **kwargs):
        if len(x.shape) != 4:
            raise ValueError(
                "input must be 4D (number_of_observations, x, y, 1)")

    def transform(self, input_data, *args, **kwargs):
        return self.extract(input_data, "sk")

    # endregion Scikit api

    def extract(self, input_data, api):
        if api == "sk":
            self.extract(input_data, "keras").numpy()
        else:
            # Pad the incoming tensor
            matpad = tf.pad(input_data, [
                [0, 0],
                [self.radius, self.radius],
                [self.radius, self.radius],
                [0, 0]
            ], mode=self.padding)

            # Sum all the pixels in the kernel
            conv = tf.nn.convolution(matpad, self.ones, 1, padding="SAME")

            # Calculate the change in shape
            mrgn_0 = int((conv.shape[1] - input_data.shape[1])/2) + self.margin
            mrgn_1 = int((conv.shape[2] - input_data.shape[2])/2) + self.margin

            # Select appropriate regions
            selection_conv = conv[:, mrgn_0:-mrgn_0, mrgn_1:-mrgn_1, :]
            selection_input = input_data[:, self.margin:-
                                         self.margin, self.margin:-self.margin, :]

            # Compute the output
            return ((selection_conv - selection_input) / selection_conv) ** self.degree


class PreprocessingBlock(Layer):
    """Preprocessing block (width, translation, zoom, seed=173)"""

    def __init__(self, width, translation, zoom, seed=173, **kwargs):
        super(PreprocessingBlock, self).__init__(**kwargs)
        self.w, self.t, self.z, self.s = width, translation, zoom, seed
        self.reshape = Reshape((28, 28, 1))
        self.width = RandomWidth(width, seed=seed, interpolation="bicubic")
        self.translation = RandomTranslation(
            *translation, seed=seed, fill_mode="constant", fill_value=0)
        self.zoom = RandomZoom(*zoom, seed=seed)
        self.resize = Resizing(28, 28, interpolation="bicubic")

    def call(self, input_data, training=None):
        x = self.reshape(input_data)

        if training:
            x = self.width(x)
            x = self.translation(x)
            x = self.zoom(x)
            x = self.resize(x)

        return x

    def get_config(self):
        config = super(PreprocessingBlock, self).get_config()
        config["width"] = self.w
        config["translation"] = self.t
        config["zoom"] = self.z
        config["seed"] = self.s

        return config

    @classmethod
    def from_config(cls, config):
        cls(**config)


class ConvBlock(Layer):
    """A block of convolutional layers"""

    def __init__(self, units, kernel_size, depth, pool=False, seed=173, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.depth = depth
        self.pool = pool
        self.seed = seed
        self.layers = []

        for i in range(depth - 1):
            self.layers.append(
                Conv2D(units, kernel_size=kernel_size, padding="same", activation="relu"))
            self.layers.append(BatchNormalization(axis=1))

        if pool:
            self.layers.append(
                Conv2D(units, kernel_size=kernel_size, padding="same", activation="relu"))
            self.layers.append(MaxPool2D())
        else:
            self.layers.append(Conv2D(
                units, kernel_size=kernel_size, padding="same", strides=2, activation="relu"))

    def call(self, input_data):
        out = input_data
        for layer in self.layers:
            out = layer(out)
        return out

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config["units"] = self.units
        config["kernel_size"] = self.kernel_size
        config["depth"] = self.depth
        config["pool"] = self.pool
        config["seed"] = self.seed

        return config

    @classmethod
    def from_config(cls, config):
        cls(**config)


# endregion custom stuff


# loading model
def get_model(seed=173):
    """Generate MSXCN"""

    x = Input((28 * 28))
    y = PreprocessingBlock(
        0.18, (0.14, 0.2), ((-0.05, -0.1), (-0.05, -0.02)), seed=seed)(x)

    # Convolutional chain
    y_0_0 = ConvBlock(32, 5, 2, True, seed=seed)(y)
    # Some papers suggest to stay away from
    y_0 = BatchNormalization(axis=1)(y_0_0)
    # using batch-norm and dropout together
    y_0 = Dropout(0.3, seed=seed)(y_0)
    # https://link.springer.com/article/10.1007/s11042-019-08453-9

    y_0 = ConvBlock(64, 5, 2, True, seed=seed)(y_0)
    y_0 = BatchNormalization(axis=1)(y_0)
    y_0 = Dropout(0.3, seed=seed)(y_0)

    y_0 = ConvBlock(128, 5, 2, True, seed=seed)(y_0)
    y_0 = GlobalAveragePooling2D()(y_0)
    y_0 = Flatten()(y_0)
    y_0 = BatchNormalization(axis=1)(y_0)
    y_0 = Dropout(0.4, seed=seed)(y_0)

    # Singularity Extractor chain 1
    y_1 = SingularityExtractor2D(6, 5, margin=1)(y)
    y_1 = LeakyReLU(0)(y_1)
    y_1 = SingularityExtractor2D(6, 5, margin=1)(y_1)
    y_1 = LeakyReLU(0)(y_1)
    y_1 = SingularityExtractor2D(6, 5, margin=1)(y_1)
    y_1 = LeakyReLU(0)(y_1)
    y_1 = Conv2D(64, kernel_size=11, padding="same", activation="relu")(y_1)
    y_1 = MaxPool2D()(y_1)
    y_1 = Flatten()(y_1)
    y_1 = BatchNormalization(axis=1)(y_1)

    # Cognitive block
    y = Concatenate()([y_0, y_1])
    y = Dense(1024)(y)
    y = LeakyReLU(0.1)(y)
    y = BatchNormalization(axis=1)(y)
    y = Dropout(0.5, seed=seed)(y)

    y = Dense(2048, activation="tanh")(y)
    y = BatchNormalization(axis=1)(y)
    y = Dropout(0.65, seed=seed)(y)

    y = Dense(10, activation="softmax")(y)

    opt = RMSprop(learning_rate=0.002, decay=0)
    model = tf.keras.Model(x, y)
    model.compile(opt, "categorical_crossentropy", metrics=["accuracy"])

    return model


# model = get_model()
# model.load_weights("..\model\checkpoint.hdf5")
# print("Model Loaded.....")


def img_preproc(data):
    """Convert an image uri into an image, grayscale, invert, resize, minmax, flatten"""
    img = Image.open(urllib.request.urlopen(data))
    img = np.invert(np.array([np.array(
        ImageOps.grayscale(
            img.resize((28, 28))
        ).getdata())
    ]).reshape(28, 28))  # /255

    return minmax_scale(img).reshape(1, 28*28)


@app.route("/", methods=["POST"])
def digit_recognizer():
    data = request.data.decode("utf-8")

    img = img_preproc(data)
    pred = 69420 # model.predict(img)
    print(img.shape)
    print(f"\n\n{pred}\n\n")

    resp = Response(str(pred))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    app.run()
