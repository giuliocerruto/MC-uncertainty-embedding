<p align="center"><img width=400 src="https://www.polito.it/images/logo_poli.svg" style="zoom:8%;" /></p>

# <p align="center">Bioinformatics (**05OVFSM** )</p>

#### A.Y. 2020/2021

 ## <p align="center">Project 9 : Uncertainty in BCNN</p>

## Borrello Simona Maria (277789, [s277789@studenti.polito.it](mailto:s277789@studenti.polito.it))

## Cerruto Giulio (277335, [giulio.cerruto@studenti.polito.it](mailto:giulio.cerruto@studenti.polito.it))

The aim of this project is to embed the MC dropout uncertainty into the learning loss of a Convolutional Neural Network, so that weight updates coming from images recognized as spurious  (i.e. when the network provides for an uncertain prediction) are reduced, while  updates coming from clear images are amplified.

The project is mainly grounded on the implementation of two new classes: an **UncertaintyDropoutModel** class, relying, in turn, on a scheduler of class **SampleWeightScheduler**.

The code relies on *Tensorflow 2.5.0* (running on eager execution, which is enabled by default in this version).

## **Uncertainty Dropout Model**

[View source](https://github.com/giuliocerruto/MC-uncertainty-embedding/blob/main/UncertaintyDropoutModel.py)

**UncertaintyDropoutModel** expands a Tensorflow Keras **Model** adding dropout layers and associating a value of uncertainty to each sample on which it is trained or tested.

Inherits From: [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

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
| **mc_replications**            | The number of times the forward pass is requested to be executed for each sample. |
| **mc_dropout_rate**            | The dropout rate of each [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layer. |
| **dropout_pos**                | Either a list of booleans of length len(underlying_model.layers)-1 specifying whether to add a dropout layer between two layers of the original model or not, or *'all'* if a dropout layer should be added anywhere. |
| **uncertainty_quantification** | A string among *'predicted_class_variances'*, *'vertical_uncertainties'* and *'entropy_uncertainties'* (see ... for more details) specifying how two quantify the uncertainty, after the MC sampled forward pass. |
| **uncertainty_tol**            | A float or double tolerance value above which to disregard a sample when updating the metrics' state. |

| **Raises**     |                                                              |
| -------------- | ------------------------------------------------------------ |
| **ValueError** | In case **dropout_pos** is not the string *'all'* or a list of bool of length len(underlying_model.layers)-1. |

This model subclasses **tf.keras.Model**, therefore, it can be configured, trained and tested with the usual and well-known methods: **model.compile()**, **model.fit()** and **model.evaluate()**.

| **Attributes**                 |                                                              |
| ------------------------------ | ------------------------------------------------------------ |
| **mc_replications**            | The number of times the forward pass is requested to be executed for each sample. |
| **dropout_rate**               | The dropout rate of each [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layer. |
| **uncertainty_quantification** | A string specifying how two quantify the uncertainty, after the MC sampled forward pass. |
| **uncertainty_function**       | A function mapping the uncertainty value of a sample to the weight it will have in the loss minimization step. The uncertainty belongs to the interval [0,1]. The function can take either one argument (the uncertainty of the sample) or two (the epoch number and the uncertainty of the sample, in this given order). |
| **layerslist**                 | The model list of layers.                                    |
| **__uncert**                   | An array collecting the sample uncertainties during the evaluation step. |
| **uncertainty_tol**            | A tolerance value above which to disregard a sample when updating the metrics' state. |
| **no_uncertainty_metrics**     | A [MetricsContainer](https://github.com/tensorflow/tensorflow/blob/fda253a68b5e55cfd388cd1d66a42ed2d29f98f3/tensorflow/python/keras/engine/compile_utils.py#L292-L557) containing the same metrics passed to **compile()**. The metrics contained are computed only on test set sample whose uncertainty measure is below **uncertainty_tol** |
| **__normalization_function**   | A function to normalize the value of uncertainty assigned to each sample, depending of the uncertainty quantification method chosen in the contructor. |
| **__test_size**                | Size of the test set passed to **evaluate()**                |
| **epoch_uncertainty_function** | A function mapping the uncertainty value of a sample to the weight it will have in the loss minimization step. The difference with **uncertainty_function** is that this one doesn't depend on the current epoch, as it is scheduled buy the **SampleWeightScheduler** |
| **sws**                        | A **SampleWeightScheduler** object scheduling which uncertainty function to use, depending on the current epoch. |

### Methods

#### **__tofunctional**

[View source](https://github.com/giuliocerruto/MC-uncertainty-embedding/blob/30c27a081bd3a6e22c2b78203281939eb0199cad/UncertaintyDropoutModel.py#L58-L65)

```python
__tofunctional(
    model
)
```

Turns a [Sequential Model](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) into a [Functional Model](https://www.tensorflow.org/guide/keras/functional), as this class works with the latter one only.

| **Args**  |                                                              |
| --------- | ------------------------------------------------------------ |
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
| **layers**      | List of layers to which new dropout layers need to be inserted. |
| **dropout_pos** | List of bool specifying whether to insert a dropout layer in a given position. |

#### __scheduler

[View source]() AGGIUNGERE LINK

```python
__scheduler(
    epoch, fun
)
```

Returns a [partial object](https://docs.python.org/3/library/functools.html#partial-objects) serving as a scheduler to the **SampleWeightScheduler** object attribute.

| **Args**  |                                                              |
| --------- | ------------------------------------------------------------ |
| **epoch** | Current epoch number.                                        |
| **fun**   | Function depending on the current epoch number and on the sample uncertainty. |

| **Returns**                                                  |
| ------------------------------------------------------------ |
| A [partial object](https://docs.python.org/3/library/functools.html#partial-objects) serving as a scheduler to the **SampleWeightScheduler** object attribute. |

