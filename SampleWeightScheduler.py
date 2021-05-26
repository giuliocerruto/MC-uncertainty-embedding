from tensorflow import keras
from keras.callbacks import Callback


class SampleWeightScheduler(Callback):

    def __init__(self, schedule, verbose=0):
        super(SampleWeightScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        fun = self.model.uncertainty_function
        self.model.epoch_uncertainty_function = self.schedule(epoch, fun)