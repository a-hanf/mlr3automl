# mlr3automl

`mlr3automl` is an AutoML package for R based on [mlr3](https://github.com/mlr-org/mlr3).  
The first version is up and running, feedback is very welcome!

Watch our “UseR! 2021” presentation on Youtube for an
introduction. The slides are [here](https://a-hanf.github.io/useR_presentation/useR_2021_mlr3automl.pdf).

[![UseR! 2021 mlr3automl](https://img.youtube.com/vi/D4MrVnumM8k/0.jpg)](https://www.youtube.com/watch?v=D4MrVnumM8k&t=86s)


## Installation

Make sure to have the latest versions of the relevant mlr3 packages.

```
devtools::install_github('https://github.com/mlr-org/mlr3extralearners')
devtools::install_github('https://github.com/a-hanf/mlr3automl', dependencies = TRUE)
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
