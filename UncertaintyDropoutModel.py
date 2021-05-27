import numpy as np
import tensorflow as tf
from tensorflow import keras
from copy import deepcopy
from inspect import signature
from functools import partial
from .SampleWeightScheduler import SampleWeightScheduler
from tensorflow.python.keras.engine import compile_utils
from scipy.stats import entropy


class UncertaintyDropoutModel(keras.Model):

    def __init__(self, input_shape, n_classes, underlying_model, uncertainty_function, MC_replications=10,
                 MC_dropout_rate=0.6, dropout_pos='all',
                 uncertainty_quantification='predicted_class_variances', uncertainty_tol=0.6):
        # togliere input_shape se non implementiamo il modo per decidere l'architettura passando una stringa a model
        super(UncertaintyDropoutModel, self).__init__()
        self.shape = input_shape
        self.n_classes = n_classes
        self.MC_replications = MC_replications
        self.dropout_rate = MC_dropout_rate
        self.uncertainty_quantification = uncertainty_quantification
        self.uncertainty_function = uncertainty_function
        self.epoch_uncertainty_function = None
        self.layerslist = list()  # list of model layers
        self.uncert = []
        self.uncertainty_tol = tf.constant([uncertainty_tol], dtype=tf.double)
        self.no_uncertainty_metrics = None
        self.__normalization_function = self.__normalize_uncertainties()
        self.__test_size = None

        if len(signature(
                self.uncertainty_function).parameters) == 1:  # if the uncertainty function has only one parameter, it won't be dependent on the epoch
            self.epoch_uncertainty_function = self.uncertainty_function
            self.sws = None
        elif len(signature(
                self.uncertainty_function).parameters) == 2:  # if it is dependent on 2 parameters, we need a schedule
            self.sws = SampleWeightScheduler(self.__scheduler)

        # converting model to functional if it is sequential
        if isinstance(underlying_model, keras.Sequential):
            underlying_model = self.__tofunctional(underlying_model)

        n_layers = len(underlying_model.layers)

        # aggiungere errore-> if self.shape!= underlying_model

        if isinstance(dropout_pos, str):
            if dropout_pos == 'all':
                dropout_pos = [True] * (n_layers - 1)

        len_dropout_pos = len(dropout_pos)

        if not all(isinstance(i, bool) for i in dropout_pos):
            raise ValueError('dropout_pos has to be an array of bool')

        if (len_dropout_pos != n_layers - 1):
            raise ValueError('dropout_pos has to be as long as the number of layers - 1')

        self.__adddropoutlayers(underlying_model.layers, dropout_pos)

    def __tofunctional(self, model):
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer

        for layer in basic_model_new.layers:
            prev_layer = layer(prev_layer)

        return keras.models.Model([input_layer], [prev_layer])

    def __adddropoutlayers(self, layers, dropout_pos):
        for idx, layer in enumerate(layers):
            self.layerslist.append(deepcopy(layer))
            if layer != layers[-1] and dropout_pos[idx]:
                self.layerslist.append(keras.layers.Dropout(rate=self.dropout_rate))

    def __scheduler(self, epoch, fun):
        return partial(fun, epoch)

    def call(self, inputs, training=True):  # implementing forward pass
        x = tf.dtypes.cast(inputs, tf.float32)

        for i, layer in enumerate(self.layerslist[1:]):
            if isinstance(layer, keras.layers.Dropout):  # if it is a dropout layer, set training parameter
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        if callbacks is not None:  # if there are more callbacks passed to the function
            if self.sws:  # append our SampleWeightScheduler if we're using it
                callbacks.append(self.sws)
            return super(UncertaintyDropoutModel, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                                                            verbose=verbose, callbacks=callbacks,
                                                            validation_split=validation_split,
                                                            validation_data=validation_data, shuffle=shuffle,
                                                            class_weight=class_weight,
                                                            sample_weight=sample_weight, initial_epoch=initial_epoch,
                                                            steps_per_epoch=steps_per_epoch,
                                                            validation_steps=validation_steps,
                                                            validation_batch_size=validation_batch_size,
                                                            validation_freq=validation_freq,
                                                            max_queue_size=max_queue_size,
                                                            workers=workers, use_multiprocessing=use_multiprocessing)
        else:
            return super(UncertaintyDropoutModel, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                                                            verbose=verbose, callbacks=self.sws,
                                                            validation_split=validation_split,
                                                            validation_data=validation_data, shuffle=shuffle,
                                                            class_weight=class_weight,
                                                            sample_weight=sample_weight, initial_epoch=initial_epoch,
                                                            steps_per_epoch=steps_per_epoch,
                                                            validation_steps=validation_steps,
                                                            validation_batch_size=validation_batch_size,
                                                            validation_freq=validation_freq,
                                                            max_queue_size=max_queue_size,
                                                            workers=workers, use_multiprocessing=use_multiprocessing)

    def __normalize_uncertainties(self):
        if self.uncertainty_quantification == 'predicted_class_variances':
            normalization_factor = 0.25
        elif self.uncertainty_quantification == 'vertical_uncertainties':
            normalization_factor = 1
        elif self.uncertainty_quantification == 'entropy_uncertainties':
            normalization_factor = 1.58496
        return (lambda x: x / normalization_factor)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred, uncert = self.__computeuncertainties(x)
            uncert = tf.map_fn(self.__normalization_function, uncert)
            sample_weight = tf.map_fn(self.epoch_uncertainty_function, uncert)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def __computeuncertainties(self, x):
        if self.uncertainty_quantification == 'predicted_class_variances':
            return self.__compute_predicted_class_variances(x)
        elif self.uncertainty_quantification == 'vertical_uncertainties':
            return self.__compute_vertical_uncertainties(x)
        elif self.uncertainty_quantification == 'entropy_uncertainties':
            return self.__compute_entropy_uncertainties(x)

    def __MC_sampling(self, x):
        p_hat = list()  # MC times predictions for the whole batch

        for _ in range(self.MC_replications):
            y_pred = self(x, training=True)  # (Batch, C)
            p_hat.append((y_pred))  # Forward pass

        p_hat = tf.stack(p_hat)  # (MC, batch, C)
        return p_hat

    def __compute_predicted_class_variances(self, x):
        p_hat = self.__MC_sampling(x)
        mean_probs = tf.math.reduce_mean(p_hat, axis=0)
        # Mean over MC samples
        predictions_uncertainty = np.argmax(mean_probs.numpy(),
                                            axis=1)  # questo è un indice della classe predetta per ogni immagine
        aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)  # prodotto tra matrici elementwise
        epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat,
                                                          axis=0) ** 2  # è un vettore in orizzontale cioè lungo c , dove c è il numero di classi
        uncertainties_among_labels = epistemic + aleatoric

        # predicted_class_variances = tf.map_fn(get_unct, uncertainties_among_labels)
        predicted_class_variances = np.asarray([uncertainty[prediction] for prediction, uncertainty in
                                                zip(predictions_uncertainty, uncertainties_among_labels)])
        return mean_probs, predicted_class_variances

    def __minmax(self, h):
        a = np.argmax(h)
        m = h[a]
        h[-a]
        d = np.array([x for i, x in enumerate(h) if i != a])
        return 1 - min(m - d)

    def __compute_vertical_uncertainties(self, x):
        p_hat = self.__MC_sampling(x)
        mean_probs = tf.math.reduce_mean(p_hat, axis=0)
        predicted_uncertanties = np.apply_along_axis(self.__minmax, 1, mean_probs.numpy())
        return mean_probs, predicted_uncertanties

    def __compute_entropy_uncertainties(self, x):
        p_hat = self.__MC_sampling(x)
        mean_probs = tf.math.reduce_mean(p_hat, axis=0)
        predicted_uncertanties = np.apply_along_axis(entropy, 1, mean_probs.numpy())
        return mean_probs, predicted_uncertanties

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        if loss is not None:
            if loss.reduction != 'none':
                raise ValueError('loss reduction has to be set to \'none\'')

        if metrics is not None:
            # self.no_uncertainty_metrics = deepcopy(metrics)
            self.no_uncertainty_metrics = compile_utils.MetricsContainer(deepcopy(metrics), weighted_metrics,
                                                                         output_names=self.output_names)

        return super(UncertaintyDropoutModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics,
                                                            loss_weights=loss_weights,
                                                            weighted_metrics=weighted_metrics, run_eagerly=run_eagerly,
                                                            steps_per_execution=steps_per_execution, **kwargs)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        self.uncert = []
        for m in self.no_uncertainty_metrics.metrics:
            m.reset_states()
        self.__test_size = len(list(x.unbatch().as_numpy_iterator()))
        return super(UncertaintyDropoutModel, self).evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose,
                                                             sample_weight=sample_weight, steps=steps,
                                                             callbacks=callbacks, max_queue_size=max_queue_size,
                                                             workers=workers, use_multiprocessing=use_multiprocessing,
                                                             return_dict=return_dict, **kwargs)

    def no_uncertainty_evaluate(self):
        bool_conditions = tf.math.less(self.uncertainty_tol, self.uncert)
        left_img = np.count_nonzero(bool_conditions)

        return {m.name: m.result() for m in self.no_uncertainty_metrics.metrics}, {
            'uncertain image percentage': left_img / self.__test_size * 100}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred, uncert = self.__computeuncertainties(x)  # training=true or false?
        uncert = tf.map_fn(self.__normalization_function, uncert)

        self.uncert = np.concatenate((self.uncert, uncert), axis=0)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        bool_conditions = tf.math.less(uncert, self.uncertainty_tol)
        y_no_uncert = tf.boolean_mask(y, bool_conditions, axis=0)

        y_pred_no_uncert = tf.boolean_mask(y_pred, bool_conditions, axis=0)

        self.no_uncertainty_metrics.update_state(y_no_uncert, y_pred_no_uncert)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def get_test_uncertainties(self):
        return self.uncert
