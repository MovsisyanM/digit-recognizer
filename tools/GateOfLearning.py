# region Imports
from keras.callbacks import Callback
import matplotlib.pyplot as plt
# endregion Imports


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
