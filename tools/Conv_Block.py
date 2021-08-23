from keras.layers import Layer, Conv2D, MaxPool2D, Dropout, BatchNormalization


class Conv_Block(Layer):
    """A block of convolutional layers"""

    def __init__(self, units, kernel_size, depth, pool=False, seed=173):
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

        self.layers.append(Dropout(0.3, seed=seed))

    def call(self, input_data):
        out = input_data
        for layer in self.layers:
            out = layer(out)
        return out
