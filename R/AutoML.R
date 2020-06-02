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
#' @section Methods
#' * `train()` \cr
#'   Runs the AutoML system. The trained model is saved in the $learner slot.
#' @import checkmate
#' @import mlr3
#' @import mlr3oml
#' @import mlr3pipelines
#' @import mlr3tuning
#' @import paradox
#' @importFrom R6 R6Class
#' @name AutoML
#' @export
#' @examples
#' "add later"
AutoMLBase <- R6Class("AutoMLBase",
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
      self$task <- task
      self$resampling <- if (!is.null(resampling)) resampling else rsmp("holdout", ratio = 0.8)
      self$tuning_terminator <- if (!is.null(terminator)) terminator else term("evals", n_evals = 10)
      self$tuner <- tnr("random_search")
    },
    train = function(row_ids = NULL) {
      self$learner$train(self$task, row_ids)
    },
    predict = function(data = NULL, row_ids = NULL) {
      if (is.null(data)) return(self$learner$predict(self$task, row_ids))
      else return(self$learner$predict(data, row_ids))
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

AutoML <- function(task, learner = NULL, resampling = NULL, measures = NULL, param_set = NULL, terminator = NULL) {
  if (class(task)[[1]] == "TaskClassif") {
    return(AutoMLClassif$new(task, learner, resampling, measures, param_set, terminator))
  } else if (class(task)[[1]] == "TaskRegr") {
    return(AutoMLRegr$new(task, learner, resampling, measures, param_set, terminator))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
