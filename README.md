<p align="center"><img width=400 src="https://www.polito.it/images/logo_poli.svg" style="zoom:8%;" /></p>

# <p align="center">Bioinformatics (**05OVFSM** )</p>

#### A.Y. 2020/2021

 ## <p align="center">Project 9 : Uncertainty in BCNN</p>

## Borrello Simona Maria (277789, [s277789@studenti.polito.it](mailto:s277789@studenti.polito.it))

## Cerruto Giulio (277335, [giulio.cerruto@studenti.polito.it](mailto:giulio.cerruto@studenti.polito.it))

The aim of this project is to embed the Monte Carlo dropout uncertainty into the learning loss of a Convolutional Neural Network, so that weight updates coming from images recognized as spurious  (i.e. when the network provides for an uncertain prediction) are reduced, while  updates coming from clear images are amplified.

The project is mainly grounded on the implementation of two new classes: an `UncertaintyDropoutModel` class, relying, in turn, on a scheduler of class `SampleWeightScheduler`.

The code relies on *Tensorflow 2.5.0* (running on eager execution, which is enabled by default in this version).

## **UncertaintyDropoutModel**

[View source](https://github.com/giuliocerruto/MC-uncertainty-embedding/blob/main/UncertaintyDropoutModel.py)

`UncertaintyDropoutModel` expands a Tensorflow Keras `Model` adding dropout layers and executing the forward pass many times. Based on this, it associates a value of uncertainty (a new one is computed at every epoch) to each sample on which it is trained or tested.  The uncertainty value gets then mapped to the weight the sample will have in the training step.

Inherits From: [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

```python
model = UncertaintyDropoutModel(underlying_model,
                                uncertainty_function,
                                mc_replications = 10,
                                mc_dropout_rate = 0.6,
                                dropout_pos = 'all',
                                uncertainty_quantification = 'predicted_class_variances',
                                uncertainty_tol=0.6)
```

| **Args**                       |                                                              |
| :----------------------------- | :----------------------------------------------------------- |
| **underlying_model**           | The [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) to build the Bayesian model on. |
| **uncertainty_function**       | A function mapping the uncertainty value of a sample to the weight it will have in the loss minimization step. The uncertainty belongs to the interval [0,1]. The function can take either one argument (the uncertainty of the sample) or two (the epoch number and the uncertainty of the sample, in this given order). |
| **mc_replications**            | The number of times the forward pass is requested to be executed at each epoch for each sample. |
| **mc_dropout_rate**            | The dropout rate of each [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layer. |
| **dropout_pos**                | Either a list of booleans of length len(underlying_model.layers)-1 specifying whether to add a dropout layer between two layers of the original model or not, or *'all'* if a dropout layer should be added anywhere. |
| **uncertainty_quantification** | A string among *'predicted_class_variances'*, *'vertical_uncertainties'* and *'entropy_uncertainties'* (see ... for more details) specifying how to quantify the uncertainty, after the *mc_replications* times repeated forward pass. |
| **uncertainty_tol**            | A float tolerance value above which to disregard a sample when updating the metrics' state (only during the evaluation step). |

| **Raises**     |                                                              |
| -------------- | ------------------------------------------------------------ |
| **ValueError** | In case **dropout_pos** is not the string *'all'* or a list of booleans of length len(underlying_model.layers)-1. |

This model subclasses `tf.keras.Model`, therefore, it can be configured, trained and tested with the usual and well-known `Model`'s methods: [`compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile), [`fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) and [`evaluate`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate).

| **Attributes**                 |                                                              |
| ------------------------------ | ------------------------------------------------------------ |
| **mc_replications**            | The number of times the forward pass is requested to be executed at each epoch for each sample. |
| **dropout_rate**               | The dropout rate of each [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layer. |
| **uncertainty_quantification** | A string specifying how to quantify the uncertainty, after the *mc_replications* times repeated forward pass. |
| **uncertainty_function**       | A function mapping the uncertainty value of a sample to the weight it will have in the loss minimization step. The uncertainty belongs to the interval [0,1]. The function can take either one argument (the uncertainty of the sample) or two (the epoch number and the uncertainty of the sample, in this given order). |
| **layerslist**                 | The model list of layers.                                    |
| **__uncert**                   | An array collecting the sample uncertainties during the evaluation step. Its length is equal to the size of the dataset passed as input to the **evaluate** method. |
| **uncertainty_tol**            | A tolerance value above which to disregard a sample when updating the metrics' state (only during the evaluation step). |
| **no_uncertainty_metrics**     | A [MetricsContainer](https://github.com/tensorflow/tensorflow/blob/fda253a68b5e55cfd388cd1d66a42ed2d29f98f3/tensorflow/python/keras/engine/compile_utils.py#L292-L557) containing the same metrics passed to the **compile** method. The metrics contained are computed only on test set samples whose uncertainty measure is below **uncertainty_tol**. |
| **__normalization_function**   | A function to normalize in the range [0,1] the value of uncertainty assigned to each sample, depending of the uncertainty quantification method chosen in the constructor. |
| **__test_size**                | Size of the test set passed to **evaluate()**.               |
| **epoch_uncertainty_function** | A function mapping the uncertainty value of a sample to the weight it will have in the loss minimization step. The difference with **uncertainty_function** is that this one doesn't depend on the current epoch, as it is scheduled by the **SampleWeightScheduler**. |
| **sws**                        | A **SampleWeightScheduler** object scheduling which uncertainty function to use, depending on the current epoch. |

### Methods

#### **__tofunctional**

[View source](https://github.com/giuliocerruto/MC-uncertainty-embedding/blob/30c27a081bd3a6e22c2b78203281939eb0199cad/UncertaintyDropoutModel.py#L58-L65)

```python
__tofunctional(
    model
)
```

Turns a [`Sequential Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) into a [`Functional Model`](https://www.tensorflow.org/guide/keras/functional), as this class only works with the latter.

| **Args**  |                                                              |
| :-------- | ------------------------------------------------------------ |
| **model** | The [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) to be turned to Functional. |

| **Returns**                            |
| -------------------------------------- |
| A model built with the Functional API. |

#### __adddropoutlayers

[View source]() AGGIUNGERE LINK

```python
__adddropoutlayers(
    layers, dropout_pos
)
```

Inserts new dropout layers to the input list of layers according to the specified positions.

| **Args**        |                                                              |
| --------------- | ------------------------------------------------------------ |
| **layers**      | List of layers of the underlying model.                      |
| **dropout_pos** | List of boolean specifying whether to insert a dropout layer in a given position. |

#### __scheduler

[View source]() AGGIUNGERE LINK

```python
__scheduler(
    epoch, fun
)
```

Returns a [`partial object`](https://docs.python.org/3/library/functools.html#partial-objects) serving as a scheduler to the `SampleWeightScheduler` object attribute.

| **Args**  |                                                              |
| --------- | ------------------------------------------------------------ |
| **epoch** | Current epoch number.                                        |
| **fun**   | Function depending on the current epoch number and on the sample uncertainty. |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| A [partial object](https://docs.python.org/3/library/functools.html#partial-objects) serving as a scheduler to the **SampleWeightScheduler** object attribute. |

#### **call**

```python
call(
    inputs, training=True
)
```

Overrides the `tf.keras.Model` [call()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#call) method. Calls the model on new inputs.

| **Args**     |                                                              |
| ------------ | ------------------------------------------------------------ |
| **inputs**   | A tensor or list of tensors.                                 |
| **training** | Boolean or boolean scalar tensor, indicating whether to run the `Network` in training mode or inference mode. |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| A tensor if there is a single output, or a list of tensors if there are more than one outputs. |

#### **fit** 

```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
```

Overrides the `tf.keras.Model` [fit()](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/training.py#L877-L1238)  method. Trains the model for a fixed number of epochs (iterations on a dataset). If the model uses the `SampleWeitghtScheduler` (i.e. If  the `uncertainty_function` of the model is dependent on 2 parameters), this method adds the `SampleWeitghtScheduler` to the callbacks list of the model. 

| **Args**                  |                                                              |
| ------------------------- | ------------------------------------------------------------ |
| **x**                     | Input data.                                                  |
| **y**                     | Target data.                                                 |
| **batch_size**            | Number of samples per gradient update.                       |
| **epochs**                | Number of epochs to train the model.                         |
| **verbose**               | Verbosity mode.                                              |
| **callbacks**             | List of callbacks to apply during training.                  |
| **validation_split**      | Fraction of the training data to be used as validation data. |
| **validation_data**       | Data on which to evaluate the loss and any model metrics at the end of each epoch. |
| **shuffle**               | Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). |
| **class_weight**          | Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). |
| **sample_weight**         | Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). |
| **initial_epoch**         | Epoch at which to start training (useful for resuming a previous training run). |
| **steps_per_epoch**       | Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. |
| **validation_steps**      | Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. |
| **validation_batch_size** | Number of samples per validation batch.                      |
| **validation_freq**       | Integer or `collections.abc.Container` instance. If an integer, specifies how many training epochs to run before a new validation run is performed. If a Container, specifies the epochs on which to run validation. |
| **max_queue_size**        | Maximum size for the generator queue.                        |
| **workers**               | Maximum number of processes to spin up when using process-based threading. |
| **use_multiprocessing**   | If `True`, use process-based threading.                      |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). |



| Raises           |                                                              |
| ---------------- | ------------------------------------------------------------ |
| **RuntimeError** | 1. If the model was never compiled or, <br />2. If model.fit is wrapped in tf.function |
| **ValueError**   | In case of mismatch between the provided input data and what the model expects or when the input data is empty. |

#### **normalize_uncertainties**

```
__normalize_uncertainties()
```

Computes the normalization factor, based on the `uncertainty_quantification` mode, passed as input.

| **Returns**                              |
| ---------------------------------------- |
| A lambda function normalizing the input. |

#### **train_step**

```python
train_step(data)
```

Overrides the `tf.keras.Model` [train_step()](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/training.py#L765-L809) method. This method contains the mathematical logic for one step of training. It includes the forward pass, loss calculation, back-propagation, and metric updates.
This method calls the private method `computeuncertainties` that computes, for each sample, a value of uncertainty. Then, the last is mapped, by `epoch_uncertainty_function`, into a weight that  will be embedded in the learning loss.
If the `sws` scheduler ( of class`SampleWeightScheduler`) is not instantiated (i.e. the weights do not depend on the current epoch), `epoch_uncertainty_function` coincides with `uncertainty_function`.



| **Args** |                                |
| -------- | ------------------------------ |
| **data** | A nested structure of Tensors. |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| A `dict` containing values that will be passed to [`tf.keras.callbacks.CallbackList.on_train_batch_end`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CallbackList#on_train_batch_end). |

#### **compute_uncertainties**

```python
__compute_uncertainties(x)
```

It computes the uncertainties associated to the input, using the chosen `uncertainty_quantification` mode. 

| **Args** |                                |
| -------- | ------------------------------ |
| **x**    | A nested structure of Tensors. |

| **Returns**         |
| ------------------- |
| A couple of Tensors |

#### mc_sampling

```python
__mc_sampling(x)
```

 Performs, for each batch, the forward pass *mc_replications* times and computes the predictions.

| **Args** |                              |
| -------- | ---------------------------- |
| **x**    | A tensor or list of tensors. |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| Tensor of shape (*mc_replications*, size of the batch, number of classes). |

#### **compute_predicted_class_variances**

```python
__compute_predicted_class_variances(x)
```

Computes, for each sample,  the mean over the *mc_replications* predicted probabilities. It also returns the predicted uncertainties, according to chosen `uncertainty_quantification`. See details below......vedere dettagli



| **Args** |                              |
| -------- | ---------------------------- |
| **x**    | A tensor or list of tensors. |

| **Returns**         |
| ------------------- |
| A couple of Tensors |

#### **min_max**

```python
__min_max(x)
```

Implements a way to calculate the dispersion in an array. It will be used in `compute_vertical_uncertainties`. 

| **Args** |                |
| -------- | -------------- |
| **x**    | A Numpy array. |

| **Returns**    |
| -------------- |
| A Numpy array. |

#### **compute_vertical_uncertainties**

```python
__compute_vertical_uncertainties(x)
```

Computes, for each sample,  the mean over the *mc_replications* predicted probabilities. It also returns the predicted uncertainties, according to chosen `uncertainty_quantification`. See details below......vedere dettagli

| **Args** |                              |
| -------- | ---------------------------- |
| **x**    | A tensor or list of tensors. |

| **Returns**         |
| ------------------- |
| A couple of Tensors |

#### **compute_entropy_uncertainties**

```python
__compute_entropy_uncertainties(x)
```

Computes, for each sample,  the mean over the *mc_replications* predicted probabilities. It also returns the predicted uncertainties, according to chosen `uncertainty_quantification`. See details below......vedere dettagli

| **Args** |                              |
| -------- | ---------------------------- |
| **x**    | A tensor or list of tensors. |

| **Returns**         |
| ------------------- |
| A couple of Tensors |

#### **compile**

```python
compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
)
```

Overrides the `tf.keras.Model` [compile()]([https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/training.py#L479-L586) method. Configures the model for training. It instantiates the `MetricsContainer` object`no_uncertainty_metrics` to evaluate the metrics only on a subset of the test samples.

| **Args**                |                                                              |
| ----------------------- | ------------------------------------------------------------ |
| **optimizer**           | String (name of optimizer) or optimizer instance.            |
| **loss**                | String (name of objective function), objective function or [`tf.keras.losses.Loss`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss) instance. |
| **metrics**             | List of metrics to be evaluated by the model during training and testing. |
| **loss_weights**        | Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. |
| **weighted_metrics**    | List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing. |
| **run_eagerly**         | Bool. Defaults to `False`. If `True`, this `Model`'s logic will not be wrapped in a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). |
| **steps_per_execution** | Int. Defaults to 1. The number of batches to run during each [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) call. |
| ** **kwargs**           | Arguments supported for backwards compatibility only.        |

| **Raises**     |                                                              |
| -------------- | ------------------------------------------------------------ |
| **ValueError** | In case of either invalid arguments for `optimizer`, `loss` , `metrics`or if *loss.reduction* is not *'none'* |

#### **evaluate**

```python
evaluate(
    x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
    callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    return_dict=False, **kwargs
)
```

Overrides the `tf.keras.Model` [evaluate()](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/training.py#L1340-L1501) method. Returns the loss value & metrics values for the model in test mode. It resets the state of `no_uncertainty_metrics`.



| **Args**                |                                                              |
| ----------------------- | ------------------------------------------------------------ |
| **x**                   | Input data.                                                  |
| **y**                   | Target data.                                                 |
| **batch_size**          | Number of samples per batch of computation.                  |
| **verbose**             | Verbosity mode.                                              |
| **sample_weight**       | Optional Numpy array of weights for the test samples, used for weighting the loss function. |
| **steps**               | Total number of steps (batches of samples) before declaring the evaluation round finished. |
| **callbacks**           | List of callbacks to apply during evaluation.                |
| **max_queue_size**      | Maximum size for the generator queue.                        |
| **workers**             | Maximum number of processes to spin up when using process-based threading. |
| **use_multiprocessing** | If `True`, use process-based threading. If unspecified, `use_multiprocessing` will default to `False`. |
| **return_dict**         | If `True`, loss and metric results are returned as a dict, with each key being the name of the metric. If `False`, they are returned as a list. |
| ** **kwargs**           | Unused at this time.                                         |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). |

| **Raises**       |                                                              |
| ---------------- | ------------------------------------------------------------ |
| **RuntimeError** | If `model.evaluate` is wrapped in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). |
| **ValueError**   | In case of invalid arguments.                                |

#### **no_uncertainty_evaluate**

```python
no_uncertainty_evaluate()
```

Calculates the percentage of *uncertain samples*, i.e. the percentage of test samples having  the value of `uncert` above `uncertainty_tol`. It also returns the results of `no_uncertainty_metrics`.

| **Returns**         |
| ------------------- |
| A couple of `dict`. |

#### **test_step**

```python
test_step(
    data
)
```

Overrides the `tf.keras.Model` [test_step()](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/engine/training.py#L1240-L1279) method. 
It identifies which samples have, during the test,  the value of `uncert` above `uncertainty_tol`. These samples are disregarded  when updating the state of `no_uncertainty_metrics`.

| **Args** |                                |
| -------- | ------------------------------ |
| **data** | A nested structure of Tensors. |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| A `dict` containing values that will be passed to [`tf.keras.callbacks.CallbackList.on_train_batch_end`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CallbackList#on_train_batch_end). |

#### **get_test_uncertainties**

```python
get_test_uncertainties()
```

It is a public method returning the uncertainties for test samples.

| **Returns**    |
| -------------- |
| A Numpy array. |

## **SampleWeightScheduler**

[View source]()

`SampleWeightScheduler` is a custom callback, that subclasses `tf.keras.callbacks.Callback` and overrides its method `on_epoch_begin`.

Inherits From: [`Callback`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback#expandable-1)

```python
model = SampleWeightScheduler(schedule, verbose=0)
```

| **Args**     |                                                              |
| ------------ | ------------------------------------------------------------ |
| **schedule** | A scheduler function.                                        |
| **verbose**  | Int. If verbose is a strictly positive number,  information are printed. |

An object of class `SampleWeightScheduler` is necessary if you want the weights update to be depended also on the index of the current epoch. 

Hence, it will be instantiated only if the number of parameters of the method `uncertainty_function` of the class `UncertaintyDropoutModel` are 2, i.e. if this function depend on 2 variables that are the index of the epoch and the value of uncertainty. 

| **Attributes** |                                                              |
| -------------- | ------------------------------------------------------------ |
| **schedule**   | A scheduler function.                                        |
| **verbose**    | Int. If verbose is a strictly positive number,  information are printed. |

### **Methods**

#### **on_epoch_begin**

```python
on_epoch_begin(epoch, logs=None)
```

Overrides the `tf.keras.callbacks.Callback` [on_epoch_begin](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/callbacks.py#L654-L665) method. 
When the method is called, i.e. at the start of an epoch, the uncertainty function is updated.

| **Args**  |                          |
| --------- | ------------------------ |
| **epoch** | Integer, index of epoch. |
| **logs**  | `Dict`                   |

## Appendix

### **uncertainty_quantification**  

This section provides details about `UncertaintyDropoutModel` [uncertainty_quantification](ffinse) attribute. inserire codice!!! 
It specifies how to quantify the uncertainty about predictions, after the *mc_replications* times repeated forward pass.

Three different modes have been implemented to perform this task. All three share the first step, i.e. the computation of the mean of predicted probabilities  over *mc_replications* run of the forward pass.  The result is, for each sample,  a *C*-dimensional array, where *C* is the number of classes.

A further explanation of each mode follows below.

#### ***predicted_class_variances***<sup>[[1](ciao)] </sup>

The uncertainty of the  *i*-th sample is  decomposed  into two parts, *aleatoric* and *epistemic* uncertainty where the former captures irreducible variability due to randomness of outcomes, and the latter, variability arising from estimation.  If *p_hat* denotes the *mc_replications* predicted probabilities, then the *aleatoric uncertainty* is computed as the mean, along MonteCarlo dimension, of the element**-**wise product of *p_hat*.  While the *epistemic uncertainty* is obtained by subtracting CONTINUARE 



#### ***vertical_uncertainties***

The uncertainty of the  *i*-th sample is obtained by computing the minimum difference between the maximum *M*  of the array  *mean_probs P(i)* and all its other entries.  Hence, the uncertainty of the  *i*-th sample is
$$
1- [\min_{c \neq j  }{( p_c(i)- M)}],\\ j = argmax(P(i)), \quad M= \max(P(i))
$$
where *P(i)* is defined as follows:
$$
P(i)=[p_c(i)]_{c \in C}
$$
 ![alt text](http://www.sciweavers.org/tex2img.php?eq=p_c%28i%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)is the mean probability  ( over *mc_replications* run of the forward pass ) that the *i*-th sample belongs to class *c* among all classes *C*.

#### *entropy_uncertainties* <sup>[[2](ciao)] </sup>

The uncertainty is measured in *entropy*, that is, for the *i*-th sample
$$
H(i)= - 	\sum_{c \in C} p_c(i)log_2[p_c(i)],
$$
where ![alt text](http://www.sciweavers.org/tex2img.php?eq=p_c%28i%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)is the mean probability  ( over *mc_replications* run of the forward pass ) that the *i*-th sample belongs to class *c* among all classes *C*.

### **uncertainty_function** 



This section provides examples for the *uncertainty_function* that maps the uncertainty value of a sample to the weight it will have in the loss minimization step. The uncertainty function should be chosen from the user. It can be more or less complex, as well as it can depend on both or just the value of the uncertainty.  Below, just few examples, obtained heuristically, (???)  have been shown: 

#### **Linear**

The *linear uncertainty function* is defined as follows: 
$$
f(uncert,epoch)= 1+ [-1.5*(epoch-1)/(N-1)]*(uncert-0.5),
$$

where *N* is the number of epochs of the model. 

<p align="center"><img width=400 src="https://github.com/giuliocerruto/MC-uncertainty-embedding/blob/main/img/linear.png" style="zoom:8%;" /></p>

#### **Exponential**

The *exponential uncertainty function* is defined as follows: 
$$
f(uncert,epoch)= [2+(epoch-N)/N]* exp(-(epoch-1)*uncert)
$$
```
f(uncert,epoch)= [2+(epoch-N)/N]* exp(-(epoch-1)*uncert)
```

where *N* is the number of epochs of the model. 

<p align="center"><img width=400 src="https://github.com/giuliocerruto/MC-uncertainty-embedding/blob/main/img/exp.jpg" style="zoom:8%;" /></p>

IMMAGINE-> METTERE SUGLI ASSI IL SIGNIFICATO

It is worth pointing out the following considerations:

* at the first epoch, all samples have the same weight in the loss minimization step. No influence neither from the value of uncertainty nor from the value of the current epoch. A such behavior seems reasonable/justifiable at the at the beginning of the training.
* because of increasing convexity, as the training continues, the uncertainty plays an increasingly central role. In fact, as  the number of the current epoch grows, the samples with low uncertainties will have matter more and more, viceversa that ones with higher uncertainties, will have matter less and less . For istance, a value of uncertainty= 0.2 is mapped to a weight = and the secondo epoch = at the last. Viceverse a value of uncertainty=0.9 is    

#todo: 

- menzionare articoli ( su dropout e epistemic varianza)

* link file excel con risultati
* successivi sviluppi
