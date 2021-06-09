import tensorflow as tf
from tensorflow import keras
from copy import deepcopy
from inspect import signature
from functools import partial
from tensorflow.python.keras.engine import compile_utils
from scipy.stats import entropy
from .SampleWeightScheduler import SampleWeightScheduler


class UncertaintyDropoutModel(keras.Model):

    def __init__(self, underlying_model, uncertainty_function, mc_replications=10,
                 mc_dropout_rate=0.6, dropout_pos='all',
                 uncertainty_quantification='predicted_class_variances', uncertainty_tol=0.6):
        
        super(UncertaintyDropoutModel, self).__init__() # calling super class constructor
        self.mc_replications = mc_replications # number of times forward pass has to be performed for each sample
        self.dropout_rate = mc_dropout_rate # dropout rate of dropout layers
        self.uncertainty_quantification = uncertainty_quantification # how to quantify uncertainty
        self.uncertainty_function = uncertainty_function # how to map uncertainty to sample weights
        self.layerslist = list()  # list of model layers
        self.__uncert = [] # array to store 
        self.uncertainty_tol = tf.constant([uncertainty_tol], dtype=tf.float32) # uncertainty tolerance
        self.no_uncertainty_metrics = None
        self.__normalization_function = self.__normalize_uncertainties() # function to normalize the uncertainty into the interval [0,1]
        self.__test_size = None # size of the test set

        if len(signature(
                self.uncertainty_function).parameters) == 1:  # if the uncertainty function has only one parameter, it won't be dependent on the epoch
            self.epoch_uncertainty_function = self.uncertainty_function
            self.sws = None
        elif len(signature(
                self.uncertainty_function).parameters) == 2:  # if it is dependent on 2 parameters, we need a schedule
            self.sws = SampleWeightScheduler(self.__scheduler) # use a sample weight scheduler in this case

        # converting model to functional if it is sequential
        if isinstance(underlying_model, keras.Sequential):
            underlying_model = self.__tofunctional(underlying_model)

        n_layers = len(underlying_model.layers) #number of layers of underlying model


        if isinstance(dropout_pos, str): #if the dropout_pos is string
            if dropout_pos == 'all': 
                dropout_pos = [True] * (n_layers - 1)
            else:
              raise ValueError('dropout_pos has to be an array of bool or the string \'all\'') # value error if dropout_pos is string, but not the string 'all' 


        len_dropout_pos = len(dropout_pos)

        if not all(isinstance(i, bool) for i in dropout_pos): #if dropout_pos contains not only boolean values
            raise ValueError('dropout_pos has to be an array of bool or the string \'all\'')

        if (len_dropout_pos != n_layers - 1): #if length of droput_pos is not appropriate 
            raise ValueError('dropout_pos has to be as long as the number of layers - 1')

        self.__adddropoutlayers(underlying_model.layers, dropout_pos) #add dropout_layers to the underlying model in positions given by dropout_pos

    def __tofunctional(self, model): #method to convert model to functional
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer

        for layer in model.layers: #loop on layers of the model 
            prev_layer = layer(prev_layer) #using output of the previous layer as the input of the next one

        return keras.models.Model([input_layer], [prev_layer]) 

    def __adddropoutlayers(self, layers, dropout_pos): #method to add dropout layers to the model
        for idx, layer in enumerate(layers):
            self.layerslist.append(deepcopy(layer)) 
            if layer != layers[-1] and dropout_pos[idx]: #if layer is not the last one and if dropout_pos contains True for that position
                self.layerslist.append(keras.layers.Dropout(rate=self.dropout_rate))

    def __scheduler(self, epoch, fun): #return a partial function
        return partial(fun, epoch) #evaluation of the function 'fun' at its first argument 'epoch'

    def call(self, inputs, training=True):  # implementing forward pass
        x = tf.dtypes.cast(inputs, tf.float32) #cast the inputs

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
            use_multiprocessing=False): #overriding the fit method of keras

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

    def __normalize_uncertainties(self): #function to set the normalization_factor according to the uncertainty_quantification
        if self.uncertainty_quantification == 'predicted_class_variances':
            normalization_factor = 0.25
        elif self.uncertainty_quantification == 'vertical_uncertainties':
            normalization_factor = 1
        elif self.uncertainty_quantification == 'entropy_uncertainties':
            normalization_factor = 1.58496
        return (lambda x: x / normalization_factor)

    def train_step(self, data): #overriding the train_step method of keras
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred, uncert = self.__compute_uncertainties(x) #compute predictions and uncertainties
            uncert = tf.map_fn(self.__normalization_function, uncert) #apply the normalization_function to uncert
            #if sws is None, epoch_uncertainty_function coincides with uncertainty_function.
            sample_weight = tf.map_fn(self.epoch_uncertainty_function, uncert) #compute the sample_weight using epoch_uncertainty_function and uncertainties
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses, sample_weight=sample_weight) #setting the sample_weight

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def __compute_uncertainties(self, x): #computing the uncertainties according to the uncertainty_quantification
        if self.uncertainty_quantification == 'predicted_class_variances':
            return self.__compute_predicted_class_variances(x)
        elif self.uncertainty_quantification == 'vertical_uncertainties':
            return self.__compute_vertical_uncertainties(x)
        elif self.uncertainty_quantification == 'entropy_uncertainties':
            return self.__compute_entropy_uncertainties(x)

    def __mc_sampling(self, x):
        p_hat = list()  # MC times predictions for the whole batch

        for _ in range(self.mc_replications):
            y_pred = self(x, training=True)  # Forward pass,  size=(Batch, C)
            p_hat.append((y_pred))  

        p_hat = tf.stack(p_hat)  # size = (mc, batch, Classes)
        return p_hat

    def __compute_predicted_class_variances(self, x):
        p_hat = self.__mc_sampling(x) #calculating the MC predictions
        mean_probs = tf.math.reduce_mean(p_hat, axis=0) # Mean over MC samples
        predictions_uncertainty = np.argmax(mean_probs.numpy(),
                                            axis=1)  #taking the index of the predicted class
        aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)  # computing aleatoric uncertainty
        epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat,
                                                          axis=0) ** 2  # computing epistemic uncertainty
        uncertainties_among_labels = epistemic + aleatoric #computing uncertainty

        predicted_class_variances = np.asarray([uncertainty[prediction] for prediction, uncertainty in
                                                zip(predictions_uncertainty, uncertainties_among_labels)])
        return mean_probs, tf.cast(predicted_class_variances, tf.float32)

    def __min_max(self, h): #this method will be used in the next one
        a = np.argmax(h) 
        m = h[a] #maximum of h
        d = np.array([x for i, x in enumerate(h) if i != a]) 
        return 1 - min(m - d)

    def __compute_vertical_uncertainties(self, x):
        p_hat = self.__mc_sampling(x) #calculating the MC predictions
        mean_probs = tf.math.reduce_mean(p_hat, axis=0) #Mean over MC samples
        predicted_uncertainties = np.apply_along_axis(self.__min_max, 1, mean_probs.numpy()) #calculating uncertainties by applying min_max
        return mean_probs, tf.cast(predicted_uncertainties, tf.float32)

    def __compute_entropy_uncertainties(self, x):
        p_hat = self.__mc_sampling(x) #calculating the MC predictions
        mean_probs = tf.math.reduce_mean(p_hat, axis=0)  #Mean over MC samples
        predicted_uncertainties = np.apply_along_axis(entropy, 1, mean_probs.numpy())  #calculating uncertainties with the entropy
        return mean_probs, tf.cast(predicted_uncertainties, tf.float32)

    def compile(self, 
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs): #overriding the method compile of keras 

        if loss is not None:
            if loss.reduction != 'none': 
                raise ValueError('loss reduction has to be set to \'none\'')

        if metrics is not None:
            self.no_uncertainty_metrics = compile_utils.MetricsContainer(deepcopy(metrics), weighted_metrics,
                                                                         output_names=self.output_names) #instantiates the `MetricsContainer` object

        return super(UncertaintyDropoutModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics,
                                                            loss_weights=loss_weights,
                                                            weighted_metrics=weighted_metrics, run_eagerly=run_eagerly,
                                                            steps_per_execution=steps_per_execution, **kwargs)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs): #overriding the evaluate method of keras
        self.__uncert = []
        for m in self.no_uncertainty_metrics.metrics:
            m.reset_states()  #It resets the state of `no_uncertainty_metrics`.

        self.__test_size = len(list(x.unbatch().as_numpy_iterator()))
        return super(UncertaintyDropoutModel, self).evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose,
                                                             sample_weight=sample_weight, steps=steps,
                                                             callbacks=callbacks, max_queue_size=max_queue_size,
                                                             workers=workers, use_multiprocessing=use_multiprocessing,
                                                             return_dict=return_dict, **kwargs)

    def no_uncertainty_evaluate(self):
        bool_conditions = tf.math.less(self.uncertainty_tol, self.__uncert) #true if the uncertainty of the sample is below the treshold
        removed_samples = np.count_nonzero(bool_conditions) #counting the number of uncert samples

        return {m.name: m.result() for m in self.no_uncertainty_metrics.metrics}, {
            'uncertain image percentage': removed_samples / self.__test_size * 100}

    def test_step(self, data): #overriding the test_step method of keras
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred, uncert = self.__compute_uncertainties(x)  #computing predictions and uncertainties on the batch
        uncert = tf.map_fn(self.__normalization_function, uncert) #applying the normalization_function to uncertainties

        self.__uncert = np.concatenate((self.__uncert, uncert), axis=0) #concatenating new uncertainties with old ones of previous batch
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses) 
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        bool_conditions = tf.math.less(uncert, self.uncertainty_tol) #true if the uncertainty of the sample is below the treshold
        y_no_uncert = tf.boolean_mask(y, bool_conditions, axis=0) #taking only certain samples

        y_pred_no_uncert = tf.boolean_mask(y_pred, bool_conditions, axis=0) #taking only predictions of certain samples

        self.no_uncertainty_metrics.update_state(y_no_uncert, y_pred_no_uncert) #updating the no_uncertainty_met

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def get_test_uncertainties(self): #method to return uncertainties of the test set
        return self.__uncert
