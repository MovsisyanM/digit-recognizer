from keras.layers import Layer, Conv2D, MaxPool2D, Dropout, BatchNormalization


class ConvBlock(Layer):
    """A block of convolutional layers"""

    def __init__(self, units, kernel_size, depth, pool=False, seed=173):
        super(ConvBlock, self).__init__()
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
        
