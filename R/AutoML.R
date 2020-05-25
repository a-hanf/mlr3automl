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
      self$learner = learner
      self$resampling = resampling
      self$measures = measures
      self$param_set = param_set
      self$tuning_terminator = terminator
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
    }
  )
)

iris <- mlr3::tsk("iris")
auto_iris <- AutoML$new(iris)
print(auto_iris)
