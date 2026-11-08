# TensorFlow-Practice

**My TensorFlow/Keras practice notebooks: tensor mechanics, regression and binary classification, worked through from scratch.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

This repo is where I taught myself the TensorFlow side of deep learning (it was called Deep-Learning before it was renamed to match what is in it). It holds four Jupyter notebooks I wrote while working through TensorFlow fundamentals, regression and classification material. It isn't a packaged product, just a record of building neural nets from the tensor level up.

Each notebook starts from raw data (or generates its own) and goes through the same loop: build a model, train it, look at where it fails, change one specific thing, and measure whether it got better. The classification notebook runs seven model variations on the same dataset to show how depth, activation functions and learning rate each move the accuracy. The work was done mostly in Google Colab, so the training runs and their outputs are saved in the notebooks. The numbers below come from those outputs. They are toy-data results from a learning exercise.

## Key Features

- **Tensor fundamentals notebook** on how TensorFlow represents and manipulates data: scalars through rank-4 tensors, `tf.constant` vs `tf.Variable`, seeded random generation (`tf.random.normal`, `tf.random.uniform`, `tf.random.shuffle`), `tf.zeros`/`tf.ones`/`tf.fill`/`tf.eye`, dtype casting, and reshaping with `reshape`, `transpose`, `expand_dims`, `squeeze` and `newaxis`.
- **Tensor math** worked by hand: matrix multiplication three ways (`tf.linalg.matmul`, the `@` operator, `tf.tensordot`), aggregations (`reduce_max/min/mean/sum`, `argmax/argmin`), variance and standard deviation through both `tf.math` and `tensorflow_probability`, plus `one_hot`, `square`, `sqrt` and `log`.
- **Device inspection**: listing physical CPU/GPU/TPU devices and a quick `nvidia-smi` to confirm the Colab GPU.
- **Regression from scratch**: build a single-neuron linear model, then turn it into a multi-layer ReLU network and watch the mean absolute error drop with each change.
- **Model experimentation loop**: training a large generated dataset across three architectures and comparing them by MAE and MSE on a held-out test set.
- **Saving and loading models** to and from `.h5`, including loading back with a custom-objects map for the metric.
- **Binary classification** on the `make_circles` dataset across seven model versions, showing why a purely linear model is stuck at ~50% and what fixes it.
- **A learning-rate-scheduler callback** that sweeps the learning rate across epochs (`1e-4 * 10**(epoch/20)`), plotted as learning rate vs loss so the best value can be read off before retraining.
- **Evaluation**: a manual train/test split, evaluation on unseen data, and a custom normalised confusion-matrix plot.
- **A medical-cost regression** on the insurance dataset, using a scikit-learn `make_column_transformer` (MinMaxScaler for numeric columns, OneHotEncoder for categoricals) in front of a Keras MLP.

## How It Works

The repo is four standalone notebooks. Each is a self-contained lesson, and together they go from "what is a tensor" to "train, tune and evaluate a network."

### 1. `Tensorflow_Fundamentals.ipynb`: the tensor layer

No models here, just the building blocks. I create tensors of increasing rank, show the difference between immutable `tf.constant` and assignable `tf.Variable`, and use seeded generators so the random results are reproducible. The second half is tensor math: matrix multiplication (checking that `tf.linalg.matmul`, `@` and `tensordot` agree), reductions and arg-extrema, statistics through `tensorflow_probability`, one-hot encoding and element-wise functions. It ends by listing the available hardware devices and running `nvidia-smi`.

### 2. `TensorFlow Regression.ipynb`: fitting a function

Starts on tiny hand-made `X`/`y` arrays so the mechanics are obvious, then scales up to a large generated dataset.

- **Baseline:** a single `Dense(1)` layer compiled with MAE loss and SGD. It underfits, sitting around MAE 9.0.
- **Improvement:** two `Dense(100, relu)` layers plus a linear output, which drops the MAE to roughly 0.22 on the small set.
- **Architecture experiments:** on the larger dataset I train three models with different depths and learning rates (Adam at 0.01) and compare them on the test split. MAE goes from ~6.0 down to ~0.52 across the three.
- **Persistence:** the best model is saved to `best_model.h5` and reloaded, with the MAE metric passed through `custom_objects` so the load works. `plot_model` draws the architecture.

### 3. `Tensorflow_Classification.ipynb`: the seven-model walkthrough

This is the most thorough notebook. The dataset is `make_circles(1000, noise=0.03)`, two interleaved rings that a linear boundary can't separate, which is the whole point.

- **Models 1–2** are linear (no hidden activations). They never get past ~50% accuracy, which confirms that a straight line can't split two rings.
- **A detour** feeds the same architecture a 1-D regression target to show the model isn't broken, it's the wrong tool for the data.
- **Models 3–4** add depth and then a **sigmoid output with ReLU hidden layers**. That is the change that makes it learn, jumping to ~98% on the circles.
- **Model 5** introduces a manual `X[:800]/X[800:]` train/test split and evaluates on the held-out 200 points (~98% test accuracy, ~0.05 test loss).
- **Model 6** attaches a `LearningRateScheduler` callback that grows the learning rate each epoch. Plotting learning rate vs loss shows where the best value is.
- **Model 7** retrains with that learning rate and reaches **~99.5% test accuracy (~0.033 loss)**.
- Finally, a custom `plot_con_mat` function draws a normalised confusion matrix from the predictions.

### 4. `Tensorflow_Regression_Medical.ipynb`: preprocessing matters

A regression on the public health-insurance dataset (age, sex, BMI, children, smoker, region → charges). The notebook downloads the CSV from a GitHub URL. I first train a plain `Dense(100)×3 → Dense(1)` ReLU network on the raw frame, then do it the better way: a scikit-learn `make_column_transformer` that MinMax-scales the numeric columns and one-hot-encodes the categoricals before the same network. The comparison is between the unscaled model and the preprocessed one. Both land around MAE $1,500–$1,700 of predicted cost, so the experiment is mostly about the preprocessing pipeline.

## Results / Highlights

- **Binary classification (`make_circles`):** ~99.5% test accuracy after tuning the learning rate with the scheduler callback, and ~98% with the default Adam setup. Linear models plateau at ~50%, which is the lesson the notebook is built around.
- **Synthetic regression:** MAE improved from ~9.0 (single linear neuron) to ~0.22 (two ReLU layers) on the small dataset, and from ~6.0 to ~0.52 across three architectures on the larger one.
- **Medical-cost regression:** MAE in the ~$1,560–$1,750 range on the insurance dataset, comparing raw and column-transformed inputs.
- **Coverage:** 4 notebooks, 116 cells in total, covering tensor ops, regression, classification, callbacks, model save/load and confusion-matrix evaluation.

## Tech Stack

- **Language:** Python 3
- **Deep learning:** TensorFlow / Keras (`tf.keras` Sequential API), `tensorflow_probability`
- **Classical ML / preprocessing:** scikit-learn (`make_circles`, `make_column_transformer`, `MinMaxScaler`, `OneHotEncoder`, `train_test_split`, `confusion_matrix`)
- **Data / numerics:** NumPy, pandas
- **Visualization:** Matplotlib, `keras.utils.plot_model`
- **Environment:** Jupyter / Google Colab (GPU runtime)

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter (or open the notebooks directly in Google Colab, where they were written)
- TensorFlow, scikit-learn, NumPy, pandas, Matplotlib and `tensorflow-probability`

### Installation
```bash
git clone https://github.com/DCode-v05/TensorFlow-Practice.git
cd TensorFlow-Practice
pip install tensorflow tensorflow-probability scikit-learn numpy pandas matplotlib jupyter
```

### Running
```bash
jupyter notebook
# then open any of the four .ipynb files and run the cells top to bottom
```

The repo has no local data files. The medical-regression notebook downloads its CSV from a public GitHub URL, so it needs an internet connection. The other notebooks generate their own data.

## Usage

Open a notebook and run it cell by cell. They are ordered to be read as well as run. Good entry points:

- New to TensorFlow? Start with `Tensorflow_Fundamentals.ipynb`.
- Want the full train → diagnose → fix → tune loop on one dataset? Read `Tensorflow_Classification.ipynb`. The seven model versions and the learning-rate sweep are the core of it.
- Want regression and preprocessing? `TensorFlow Regression.ipynb` and `Tensorflow_Regression_Medical.ipynb`.

To reuse a trained model, the regression notebook shows the `.h5` save/load pattern (note the `custom_objects` map when loading a model that uses a custom metric name).

## Project Structure

```
TensorFlow-Practice/
├── Tensorflow_Fundamentals.ipynb        # Tensor creation, math, stats, device check
├── TensorFlow Regression.ipynb          # Linear -> MLP regression, model experiments, save/load
├── Tensorflow_Classification.ipynb      # make_circles, 7 model versions, LR scheduler, confusion matrix
├── Tensorflow_Regression_Medical.ipynb  # Insurance-cost regression with column transformers
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
