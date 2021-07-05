mlr3automl
================

## Installation

``` r
devtools::install_github('https://github.com/mlr-org/mlr3extralearners')
devtools::install_github('https://github.com/a-hanf/mlr3automl', dependencies = TRUE)
```

## Creating a model

`mlr3automl` creates classification and regression models on tabular
data sets. The entry point is the `AutoML` function, which requires an
`mlr3::TaskClassif` or `mlr3::TaskRegr` object. Creating a model on the
`iris` data set works like this:

``` r
library(mlr3)
library(mlr3automl)

iris_task = tsk("iris")
iris_model = AutoML(iris_task)
```

This creates a stable Machine Learning pipeline with Logistic
Regression, Random Forest and Gradient Boosting models. For supplying
your own data sets, convert them into into an [mlr3
Task](https://mlr3book.mlr-org.com/tasks.html).

## Training, predictions and resampling

In the above code, our model has not been trained yet. Model training is
very similar to [training and predictions in
mlr3](https://mlr3book.mlr-org.com/train-predict.html). We use the
`train()` method, which takes a vector of training indices as an
additional argument.

``` r
train_indices = sample(1:iris_task$nrow, 2/3*iris_task$nrow)
iris_model$train(row_ids = train_indices)
```

To make predictions we use the `predict()` method. We can supply a
held-out data set here, or alternatively use indices to specify
observations that were not used during training.

``` r
predict_indices = setdiff(1:iris_task$nrow, train_indices)
predictions = iris_model$predict(row_ids = predict_indices)
```

In order to make this a bit more convenient, `mlr3automl` also provides
a resampling endpoint. Instead of performing the train-test-split
ourselves, we can let `mlr3automl` do this for us.

    resampling_result = iris_model$resample()

We obtain an `mlr3::ResampleResult`. For more information on how to
analyze this result, see the [resampling section in the
mlr3book](https://mlr3book.mlr-org.com/resampling.html).

## Customisation

`mlr3automl` offers the following customization options in the `AutoML`
function:

-   custom runtime budgets:
    `model = AutoML(task = iris_task, runtime = 180)`
-   custom [measures](https://mlr3book.mlr-org.com/list-measures.html):
    `model = AutoML(task = iris_task, measure = msr("classif.logloss"))`
-   custom [learners](https://mlr3book.mlr-org.com/learners.html):
    `model = AutoML(task = iris_task, learner_list = c("classif.svm", "classif.ranger"))`
-   custom
    [resampling](https://mlr3book.mlr-org.com/resampling.html#resampling):
    `model = AutoML(task = iris_task, resampling = rsmp("cv"))`
-   custom preprocessing - choose “none,” “stability” or “full” for
    predefined pipelines:
    `model = AutoML(task = iris_task, learner_list = c("classif.ranger"), preprocessing = "none")`
-   custom preprocessing - supply your own
    [Graph](https://mlr3book.mlr-org.com/pipe-nodes-edges-graphs.html):
    see example \#2 below.
-   custom parameter spaces and parameter transformations - see example
    \#3 below.

## Examples

Example \#1: we create a regression model with custom learners and a
fixed time budget. Every hyperparameter evaluation is stopped after 10
seconds. After 300 seconds of training, tuning is stopped and the best
result obtained so far is returned:

``` r
automl_model = AutoML(
  task=tsk("mtcars"),
  learner_list=c("regr.ranger", "regr.lm"),
  learner_timeout=10,
  runtime=300)
```

Example \#2: we create a pipeline of preprocessing operators using
`mlr3pipelines`. This pipeline replaces the preprocessing pipeline in
`mlr3automl`. You can tune the choice of custom preprocessing operators
and their associated hyperparameters by supplying additional parameters
(see example \#3).

``` r
library(mlr3pipelines)
imbalanced_preproc = po("imputemean") %>>%
  po("smote") %>>%
  po("classweights", minor_weight=2)

automl_model = AutoML(task=tsk("pima"),
  preprocessing = imbalanced_preproc) 
```

Example \#3: we add a k-nearest-neighbors classifier to the learners,
which has no pre-defined hyperparameter search space in `mlr3automl`. To
perform hyperparameter tuning, we supply a parameter set using
`paradox`. A parameter transformation is supplied in order to sample the
hyperparameter on an exponential scale.

``` r
library(paradox)
new_params = ParamSet$new(list(
  ParamInt$new("classif.kknn.k",
    lower = 1, upper = 5, default = 3, tags = "kknn")))
    
my_trafo = function(x, param_set) {
  if ("classif.kknn.k" %in% names(x)) {
    x[["classif.kknn.k"]] = 2^x[["classif.kknn.k"]]
    }
    return(x)
}
    
automl_model = AutoML(
  task=tsk("iris"), 
  learner_list="classif.kknn",
  additional_params=new_params,
  custom_trafo=my_trafo)
```

## Background

`mlr3automl` tackles the challenge of Automated Machine Learning from
multiple angles.

### Flexible preprocessing using [mlr3pipelines](https://mlr3book.mlr-org.com/pipelines.html)

We tested `mlr3automl` on 39 challenging data sets in the [AutoML
Benchmark](https://openml.github.io/automlbenchmark/automl_overview.html).
By including up to 12 preprocessing steps, `mlr3automl` is stable in the
presence of missing data, categorical and high cardinality features,
huge data sets and constrained time budgets.

### Strong and stable learning algorithms

We evaluated many learning algorithms and implementations in order to
find the most stable and accurate learners for `mlr3automl`. We decided
to use the following:

-   Logistic Regression (Fan et al. 2008), which works best for small
    data sets or when the number of features is very high
-   Random Forest (Wright and Ziegler 2017) for its high performance,
    fast and highly parallelised training
-   Gradient Boosting (Chen and Guestrin 2016) for the same reasons as
    Random Forest

### Static portfolio of known good pipelines

When selecting the best model for a data set, up to 8 predefined
pipelines are evaluated first. These are our most robust, fast and
accurate pipelines, which provide us with a strong baseline even on the
most challenging tasks.

### Hyperparameter Optimization with [Hyperband](https://mlr3hyperband.mlr-org.com/)

We use Hyperband (Li et al. 2017) to tune the hyperparameters of our
machine learning pipeline. Hyperband speeds up random search through
adaptive resource allocation and early-stopping. When tuning the
hyperparameters in `mlr3automl`, at first many learners will be
evaluated on small subsets of the dataset (this is quick). Later on,
fewer models get trained on larger subsets or the full dataset (which is
more expensive computationally). This allows us to find promising
pipelines with little computational cost.

## Performance and benchmarks

| Framework        | avg. rank(binary tasks) | avg. rank(multi-class) | Failures |
|:-----------------|:------------------------|:-----------------------|---------:|
| AutoGluon        | 2.32                    | 2.09                   |        0 |
| autosklearn v0.8 | 3.34                    | 3.15                   |        2 |
| autosklearn v2.0 | 3.57                    | 4.06                   |        9 |
| H2O AutoML       | 3.18                    | 3.24                   |        2 |
| mlr3automl       | 4.55                    | 3.79                   |        0 |
| TPOT             | 4.05                    | 4.68                   |        6 |

We benchmarked mlr3automl on the AutoML benchmark, which contains 39
challenging classification data sets. Under a restrictive time budget of
at most 1 hour per task, mlr3automl successfully completed every single
task.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-chen2016xgboost" class="csl-entry">

Chen and Guestrin. 2016. “Xgboost: A Scalable Tree Boosting System.”
*Proceedings of the 22nd Acm Sigkdd International Conference on
Knowledge Discovery and Data Mining*, 785–94.

</div>

<div id="ref-fan2008liblinear" class="csl-entry">

Fan et al. 2008. “LIBLINEAR: A Library for Large Linear Classification.”
*The Journal of Machine Learning Research* 9 (August): 1871–74.

</div>

<div id="ref-li2017hyperband" class="csl-entry">

Li et al. 2017. “Hyperband: A Novel Bandit-Based Approach to
Hyperparameter Optimization.” *The Journal of Machine Learning Research*
18 (1): 6765–6816.

</div>

<div id="ref-JSSv077i01" class="csl-entry">

Wright and Ziegler. 2017. “Ranger: A Fast Implementation of Random
Forests for High Dimensional Data in C++ and R.” *Journal of Statistical
Software, Articles* 77 (1): 1–17.
<https://doi.org/10.18637/jss.v077.i01>.

</div>

</div>
