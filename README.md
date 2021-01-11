# mlr3automl

In this repository we are developing `mlr3automl`, an AutoML package for mlr3.  
The first version is up and running, feedback is very welcome!

## Installation

Make sure to have the latest versions of the relevant mlr3 packages.

```
devtools::install_github('https://github.com/mlr-org/mlr3@master')
devtools::install_github('https://github.com/mlr-org/mlr3tuning@autotuner-notimeout')
devtools::install_github('https://github.com/mlr-org/mlr3extralearners@master')
devtools::install_github('https://github.com/a-hanf/mlr3automl@master', dependencies = TRUE)
```

## Using mlr3automl

You can create your AutoML learner by passing a classification or regression [Task](https://mlr3book.mlr-org.com/tasks.html) from mlr3.

```
iris_task <- tsk('iris')
model <- AutoML(iris_task)
model$train()
```

## Documentation

[The vignette](vignettes/mlr3automl.md) is a good starting point.

For the function reference, check the roxygen documentation:
```
?mlr3automl::AutoML
```
