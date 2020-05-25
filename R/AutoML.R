#' @title AutoML
#' @format [R6Class] AutoML
#' @usage NULL
#' @format [`R6Class`].
#' @description
#' base class for AutoML in mlr3automl. Has subclasses for Classification and Regression.
#' @section Construction:
#' ```
#' AutoML$new(task)
#' ```
#' @section Internals:
#' The AutoML class uses `mlr3pipelines` to create a GraphLearner. Then other stuff happens
#' @section Fields:
#' * `task` :: `Task` object from `mlr3` \cr
#'   Contains the data and some meta-features (like the target variable)
#' * `learner` :: `GraphLearner` object from `mlr3pipelines` \cr
#'   Contains the machine learning pipeline containing preprocessing steps and a `Learner` object
#' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
#'   Contains the resampling method to be used for hyper-parameter optimization
#' * `measures` :: `Measure` object from `mlr_measures` \cr
#'   Contains the performance measure, for which we optimize during training
#' * `param_set` :: `ParamSet` object from `paradox` \cr
#'   Contains the parameter space over which we optimize
#' * `tuning_terminator` :: `Terminator` object from `mlr3tuning` \cr
#'   Contains the termination criterion for model tuning
#' * `tuner` :: `Tuner` object from `mlr3tuning` \cr
#'   Contains the tuning strategy used for hyper-parameter optimization (default: Random Search)
#' * `train_performance` :: numeric: performance of trained model \cr
#'   This is the model performance during training with the chosen resampling strategy
#' @section Methods
#' * `train()` \cr
#'   Runs the AutoML system. The trained model is saved in the $learner slot.
#' @import checkmate
#' @import mlr3
#' @import mlr3pipelines
#' @import mlr3tuning
#' @import paradox
#' @importFrom R6 R6Class
#' @name AutoML
#' @export
#' @examples
#' 'add later'
AutoML = R6Class("AutoML",
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
