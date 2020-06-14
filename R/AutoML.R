#' @title AutoML
#' @format [R6Class] AutoML
#' @usage NULL
#' @format [`R6Class`].
#' @description
#' base class for AutoML in mlr3automl. Has subclasses for Classification and Regression.
#' @section Construction:
#' ```
#' AutoMLBase$new(task)
#' ```
#' @section Internals:
#' The AutoML class uses `mlr3pipelines` to create a machine learning pipeline.
#' This pipeline contains multiple models (decision tree, random forest, XGBoost),
#' which are wrapped in a GraphLearner. This GraphLearner is wrapped in an
#' AutoTuner for Hyperparameter Optimization and during training or resampling.
#' @section Fields:
#' * `task` :: `Task` object from `mlr3` \cr
#'   Contains the data and some meta-features (like the target variable)
#' * `learner` :: `GraphLearner` object from `mlr3pipelines` \cr
#'   Contains the machine learning pipeline with preprocessing and multiple learners
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
#'   Note: We will be switching to Hyperband for the tuning as soon as it works.
#' @section Methods
#' * `train()` \cr
#'   Runs the AutoML system. The trained model is saved in the $learner slot.
#' * `predict(data = NULL, row_ids = NULL)` \cr
#'   `data.frame | data.table | Task -> PredictionClassif or PredictionRegr`
#'   Returns a Prediction object for the given data based on the trained model.
#'   If data is NULL, defaults to the task used for training
#'   `resample(outer_resampling_holdout_ratio = 0.8)`
#'   `double(1) -> ResampleResult`
#'   Performs nested resampling with a train/test split as the outer resampling
#' @import checkmate
#' @import mlr3
#' @import mlr3learners
#' @import mlr3oml
#' @import mlr3pipelines
#' @import mlr3tuning
#' @import paradox
#' @importFrom R6 R6Class
#' @export
#' @name AutoMLBase
#' @examples
#' "add later"
AutoMLBase = R6Class("AutoMLBase",
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
      assert_task(task)
      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measures)) assert_measures(measures)
      if (!is.null(param_set)) assert_param_set(param_set)
      # FIXME: find / write assertion for terminator class
      # if (!is.null(terminator)) assert_terminator(terminator)
      self$task = task
      self$resampling = resampling %??% rsmp("holdout", ratio = 0.8)
      self$tuning_terminator = terminator %??% term("evals", n_evals = 10)
      self$tuner = tnr("random_search")
    },
    train = function(row_ids = NULL) {
      self$learner$train(self$task, row_ids)
    },
    predict = function(data = NULL, row_ids = NULL) {
      if (is.null(data)) {
        return(self$learner$predict(self$task, row_ids))
      } else {
        return(self$learner$predict(data, row_ids))
      }
    },
    resample = function(outer_resampling_holdout_ratio = 0.8) {
      outer_resampling = rsmp("holdout", ratio = outer_resampling_holdout_ratio)
      resample_result = mlr3::resample(self$task, self$learner,
                                       outer_resampling, store_models = TRUE)
      self$learner = resample_result$learners[[1]]
      return(resample_result)
    }
  )
)

#' @title Interface function for mlr3automl
#'
#' @description
#' Interface for classes AutoMLClassif and AutoMLRegr.
#' @inheritParams AutoMLBase
#' @return ['AutoMLClassif' | 'AutoMLRegr']
#' @export
#' @examples
#' \dontrun{
#' automl_object = AutoML(tsk("iris"))
#' }
AutoML = function(task, learner = NULL, resampling = NULL, measures = NULL,
                   param_set = NULL, terminator = NULL) {
  if (class(task)[[1]] == "TaskClassif") {
    task$col_roles$stratum = task$col_info$id[task$col_info$type == "factor"]
    return(AutoMLClassif$new(task, learner, resampling,
                             measures, param_set, terminator))
  } else if (class(task)[[1]] == "TaskRegr") {
    task$col_roles$stratum = task$col_info$id[task$col_info$type == "factor"]
    return(AutoMLRegr$new(task, learner, resampling,
                          measures, param_set, terminator))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
