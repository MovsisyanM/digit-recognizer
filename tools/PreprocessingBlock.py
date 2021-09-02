from keras.layers import Layer, Reshape
from keras.layers.experimental.preprocessing import RandomWidth, RandomTranslation, RandomZoom, Resizing


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
