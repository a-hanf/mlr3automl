library(R6)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3)
library(paradox)

AutoML = R6Class(
  "AutoML",
  public = list(
    task = NULL,
    learner = NULL,
    resampling = NULL,
    measures = NULL,
    param_set = NULL,
    tuning_terminator = NULL,
    tuner = NULL,
    initialize = function(task, learner = NULL, resampling = NULL,
      measures = NULL, param_set = NULL, terminator = NULL) {
      private$..perform_input_checks(task, learner, resampling, measures, param_set, terminator)
      self$task = task
      self$learner = if (!is.null(learner)) learner else private$..get_default_learner()
      self$resampling = if (!is.null(resampling)) resampling else rsmp("holdout", ratio = 0.8)
      self$measures = if (!is.null(measures)) measures else mlr_measures$get("classif.acc")
      self$param_set = if (!is.null(param_set)) param_set else ParamSet$new(list(
        ParamDbl$new("classif.rpart.cp", lower = 0.001, upper = 0.1),
        ParamInt$new("classif.rpart.minsplit", lower = 1, upper = 10)
      ))
      self$tuning_terminator = if (!is.null(terminator)) terminator else term("clock_time", secs = 10)
      self$tuner <- tnr("random_search")
    },
    train = function() {
      tuning <- TuningInstance$new(
        task = self$task,
        learner = self$learner,
        resampling = self$resampling,
        measures = self$measures,
        param_set = self$param_set,
        terminator = self$tuning_terminator
      )
      self$tuner$tune(tuning)
      self$learner$param_set$values = tuning$result$params
    }
  ),
  private = list(
    ..perform_input_checks = function(task, learner, resampling, measures, param_set, terminator) {
      assert_task(task)
      if (!is.null(learner)) assert_learner(learner)
      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measures)) assert_measures(measures)
      if (!is.null(param_set)) assert_param_set(param_set)
      # FIXME: find / write assertion for terminator class
      # if (!is.null(terminator)) assert_terminator(terminator)
    },
    ..get_default_learner = function() {
      pipeline = po("imputemedian") %>>%
        po("learner", learner = mlr_learners$get("classif.rpart"))
      return(GraphLearner$new(pipeline))
    }
  )
)

iris <- mlr3::tsk("iris")
auto_iris <- AutoML$new(iris)
best_model <- auto_iris$train()
