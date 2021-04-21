## Objective

To explore and discover any risk factors that might likely contribute to late payments or non-payments.

## Scenario

ViaBill allows customers to purchase an item, and pay for it in 4 equal installments over the next 4 weeks. Most customers will pay these required installments on time. However, some customers will pay them late and some customers will unfortunately dishonor their agreement (default) and not pay all the installments.
By exploring the synthetic data, can you find identify factors which may be used to predict which customers will pay late?
And even more importantly, can you identify factors which may be used to predict which customers will not pay an installment at all?

## Overview

This is Kedro project [Kedro documentation](https://kedro.readthedocs.io).
It contains Exploratory Data Analysis and solution developed in jupyter notebooks (./notebooks) and deployed as a kedro data transformation pipelines (./src/viabill/pipeline).

Pippen's schema can be seen by running `kedro viz`.

It also utilize MlFlow experiment tracking.

## How to install dependencies

To install dependencies, run:

```
conda env create --file src/environment.yml && \
conda activate viabill && \
kedro install
```

run mlflow init to setup mlflow
```
kedro mlflow init
```

## How to run Kedro pipeline

You can run Kedro project with:

```
kedro run
```

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `kedro install` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```
