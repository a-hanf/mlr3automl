# mlr3automl - under development

In this repository we are developing `mlr3automl`, an AutoML package for mlr3.  
The project started in April 2020 and is supposed to be working in October 2020.

## Installation

`devtools::install_github('https://github.com/a-hanf/mlr3automl')`

## Using mlr3automl

You can create your AutoML learner by passing a classification or regression [Task](https://mlr3book.mlr-org.com/tasks.html) from mlr3.

```
iris_task <- tsk('iris')
model <- AutoML(iris_task)
model$train()
```
