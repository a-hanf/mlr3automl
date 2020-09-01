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
#' * `learner_list` :: `List` of names for `mlr3 Learners` \cr
#'   Can be used to customize the learners to be tuned over. If no parameter space
#'   is defined for the selected learner, it will be run with default parameters.
#'   Might break mlr3automl if the learner is incompatible with the provided task
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
    learner_list = NULL,
    learner = NULL,
    resampling = NULL,
    measures = NULL,
    param_set = NULL,
    tuning_terminator = NULL,
    tuner = NULL,
    encapsulate = NULL,
    initialize = function(task, learner_list = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL,
                          terminator = NULL, encapsulate = TRUE) {
      assert_task(task)
      for (learner in learner_list) {
        expect_true(learner %in% mlr_learners$keys())
      }
      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measures)) assert_measures(measures)
      if (!is.null(param_set)) assert_param_set(param_set)
      # FIXME: find / write assertion for terminator class
      # if (!is.null(terminator)) assert_terminator(terminator)
      self$task = task
      self$resampling = resampling %??% rsmp("holdout")
      self$tuning_terminator = terminator %??% trm('evals', n_evals = 10)
      self$tuner = tnr("random_search")
      self$encapsulate = encapsulate
    },
    train = function(row_ids = NULL) {
      self$learner$train(self$task, row_ids)
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
      }
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
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
      }
      return(resample_result)
    },
    tuned_params = function() {
      return(self$learner$tuning_instance$archive$best())
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
AutoML = function(task, learner_list = NULL, resampling = NULL, measures = NULL,
                   param_set = NULL, terminator = NULL, encapsulate = TRUE) {
  if (class(task)[[1]] == "TaskClassif") {
    target_is_factor = task$col_info[task$col_info$id == task$target_names, ]$type == "factor"
    if (length(target_is_factor) == 1 && target_is_factor) {
      task$col_roles$stratum = task$target_names
    }
    return(AutoMLClassif$new(task, learner_list, resampling, measures,
                             param_set, terminator, encapsulate))
  } else if (class(task)[[1]] == "TaskRegr") {
    task$col_roles$stratum = task$col_info$id[task$col_info$type == "factor"]
    return(AutoMLRegr$new(task, learner_list, resampling, measures,
                          param_set, terminator, encapsulate))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
