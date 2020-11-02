#' @title Interface function for mlr3automl
#'
#' @description
#' Interface for classes AutoMLClassif and AutoMLRegr.
#' @param task
#' * `task` :: `Task` object from `mlr3` \cr
#'   Contains the task to be solved.
#' @param learner_list
#' * `learner_list` :: `List` of names for `mlr3 Learners` \cr
#'   Can be used to customize the learners to be tuned over. If no parameter space
#'   is defined for the selected learner, it will be run with default parameters.
#'   Default learners for classification: `c("classif.ranger", "classif.xgboost", "classif.liblinear")`,
#'   default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")`.
#'   Might break mlr3automl if the learner is incompatible with the provided task.
#' @param learner_timeout
#' * `learner_timeout` :: `Integer` \cr
#'   Budget (in seconds) for a single learner during training of the pipeline.
#'   If this budget is exceeded, the learner is replaced with the fallback
#'   learner (`lrn("classif.featureless")` or `lrn("regr.featureless")`).
#' @param resampling
#' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
#'   Contains the resampling method to be used for hyper-parameter optimization.
#'   Defaults to `rsmp("holdout")`.
#' @param measure
#' * `measure` :: `Measure` object from `mlr_measures` \cr
#'   Contains the performance measure, for which we optimize during training.
#'   Defaults to `msr("classif.acc")` for classification and `msr("regr.rmse")`
#'   for regression.
#' @param terminator
#' * `terminator` :: `Terminator` object from `mlr3tuning` \cr
#'   Contains the termination criterion for model tuning. Note that the Hyperband
#'   tuner might stop training before the budget is exhausted.
#'   Defaults to `trm("none")`
#' @return ['AutoMLClassif' | 'AutoMLRegr']
#' Returned class depends on the type of task.
#' @export
#' @examples
#' \dontrun{
#' automl_object = AutoML(tsk("iris"))
#' }
AutoML = function(task, learner_list = NULL, learner_timeout = NULL,
                  resampling = NULL, measure = NULL, terminator = NULL) {
  if (task$task_type == "classif") {
    # stratify target variable so that every target label appears
    # in all folds while resampling
    target_is_factor = task$col_info[task$col_info$id == task$target_names, ]$type == "factor"
    if (length(target_is_factor) == 1 && target_is_factor) {
      task$col_roles$stratum = task$target_names
    }
    return(AutoMLClassif$new(task, learner_list, learner_timeout,
                             resampling, measure, terminator))
  } else if (task$task_type == "regr") {
    return(AutoMLRegr$new(task, learner_list, learner_timeout,
                          resampling, measure, terminator))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
