# Deep-Learning

Four TensorFlow/Keras practice notebooks: tensor basics, regression, binary classification and a regression on a real dataset. I wrote them while working through TensorFlow fundamentals, mostly on Google Colab.

## Notebooks

- `Tensorflow_Fundamentals.ipynb`: tensors of different rank, `tf.constant` vs `tf.Variable`, seeded random tensors, matrix multiplication, aggregations, and a check of the available CPU/GPU devices.
- `TensorFlow Regression.ipynb`: regression on hand-made and generated data. A single-neuron baseline, then a ReLU network, then three architectures compared on a held-out set, and saving and loading a model as `.h5`.
- `Tensorflow_Classification.ipynb`: binary classification on scikit-learn's `make_circles`. A linear model stays near 50% accuracy; ReLU hidden layers with a sigmoid output get to about 98%. A `LearningRateScheduler` sweep picks a learning rate, and a retrained model reaches about 99.5% test accuracy. Ends with a normalised confusion matrix.
- `Tensorflow_Regression_Medical.ipynb`: predicting insurance charges from the public insurance dataset (downloaded from a GitHub URL inside the notebook), comparing raw inputs against a `make_column_transformer` with `MinMaxScaler` and `OneHotEncoder`.

The numbers above come from the notebook outputs. They are toy-data results from a learning exercise.

## Running

```bash
pip install tensorflow tensorflow-probability scikit-learn pandas numpy matplotlib jupyter
jupyter notebook
```

The notebooks need no local data files. `Tensorflow_Regression_Medical.ipynb` needs internet access for the dataset.
