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
    train_performance = NULL,
    initialize = function(task, learner = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL) {
      private$..perform_input_checks(task, learner, resampling, measures, param_set, terminator)
      self$task = task
      # get_default_learner should be implemented by child classes
      self$learner = if (!is.null(learner)) learner else private$..get_default_learner()
      self$resampling = if (!is.null(resampling)) resampling else rsmp("holdout", ratio = 0.8)
      self$tuning_terminator = if (!is.null(terminator)) terminator else term("evals", n_evals = 5)
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
      self$train_performance = tuning$result$perf
    }
  ),
  private = list(
    ..perform_input_checks = function(task, learner, resampling, measures, param_set, terminator) {
      assert_task(task)
      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measures)) assert_measures(measures)
      if (!is.null(param_set)) assert_param_set(param_set)
      # FIXME: find / write assertion for terminator class
      # if (!is.null(terminator)) assert_terminator(terminator)
    }
  )
)
