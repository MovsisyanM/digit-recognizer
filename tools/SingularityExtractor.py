# region Imports
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np
# endregion Imports


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

        super(SingularityExtractor2D, self).__init__(**kwargs)

    # region Keras api

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("input must be 4D (batch_size, x, y, channels)")

        super(SingularityExtractor2D, self).build(input_shape)

        # Lazy load conv kernel
        # Hard-coded 1-channel for the digit-recognizer project,
        # will (hopefully) change after
        # , dtype=self.dtype)
        self.ones = tf.ones((self.kernel_size, self.kernel_size, 1, 1))

    def get_config(self):
        """Layer to dict = serializability"""

        config = super(SingularityExtractor2D, self).get_config()
        print(config)
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

    def compute_output_shape(self, input_shape):
        return (input_shape[-3], input_shape[-2] - 2 * self.margin, input_shape[-1] - 2 * self.margin, 1)

    def compute_output_signature(self, input_spec):
        shape = (input_spec.shape[-3], input_spec.shape[-2] - 2 *
                 self.margin, input_spec.shape[-1] - 2 * self.margin, 1)
        return tf.TensorSpec(shape, self.dtype)

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
