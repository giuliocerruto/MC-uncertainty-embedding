# Bioinformatics

 ## <span style="color:blue">Project 9 : Uncertainty in BCNN </span>

## <span style="color:blue">Borrello Simona Maria (277789)</span>

## <span style="color:blue">Cerruto Giulio (277335)</span>

The aim of this project is to embed the MC dropout uncertainty into the learning loss of a Convolutional Neural Network, so that weight updates coming from images recognized as spurious  (i.e. when the network provides for an uncertain prediction) are reduced, while  updates coming from clear images are amplified.

The project is mainly grounded on the implementation of a new class: <span style="background-color: #ffecb8">**UncertaintyDropoutModel**</span>, relying, in turn, on a scheduler of class <span style="background-color: #ffecb8">**SampleWeightScheduler**</span>.

The code relies on *Tensorflow 2.5.0* (running on eager execution, which is enabled by default in this version).

## **Uncertainty Dropout Model**

<span style="background-color: #ffecb8">**UncertaintyDropoutModel**</span> expands a Tensorflow Keras **Model** adding dropout layers and associating a value of uncertainty to each sample on which it is trained or tested.

Inherits From: <span style="background-color: #ffecb8">[Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)</span>

```python
model = UncertaintyDropoutModel(underlying_model,
                                ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)'uncertainty_function',
                                MC_replications = 10,
                                MC_dropout_rate = 0.6,
                                dropout_pos = 'all',
                                uncertainty_quantification = 'predicted_class_variances',
                                uncertainty_tol = 0.6)
```

| **Args**                       |                                                              |
| :----------------------------- | :----------------------------------------------------------- |
| **underlying_model**           | The [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) to build the Bayesian model on. |
| **uncertainty_function**       | A function mapping the uncertainty value of a sample to the weight it will have in the loss minimization step. The uncertainty belongs to the interval [0,1]. The function can take either one argument (the uncertainty of the sample) or two (the epoch number and the uncertainty of the sample, in this given order). |
| **MC_replications**            | The number of times the forward pass is requested to be executed for each sample. |
| **MC_dropout_rate**            | The dropout rate of the [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layer. |
| **dropout_pos**                | Either a list of booleans of length len(underlying_model.layers)-1 specifying whether to add a dropout layer between two layers of the original model or not, or '*all*' if a dropout layer should be added anywhere. |
| **uncertainty_quantification** | A string specifying how two quantify the uncertainty, after the MC sampled forward pass. |
| **uncertainty_tol**            | A tolerance value above which disregard a sample when updating the metrics' state. |









