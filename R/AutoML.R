#' @title Interface function for mlr3automl
#'
#' @description
#' Creates an instance of [AutoMLClassif][mlr3automl::AutoMLClassif] or [AutoMLRegr][mlr3automl::AutoMLRegr].
#'
#' @param task ([`Task`][mlr3::Task]) \cr
#' Contains the task to be solved. Currently [`TaskClassif`][mlr3::TaskClassif] and [`TaskRegr`][mlr3::TaskRegr] are supported.
#' @param learner_list (`list()` | `character()`) \cr
#' `List` of names from [mlr_learners][mlr3::mlr_learners]. Can be used to customize the learners to be tuned over. \cr
#' Default learners for classification: `c("classif.ranger", "classif.xgboost", "classif.liblinear")` \cr
#' Default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")` \cr
#' Might break mlr3automl if a user-provided learner is incompatible with the provided task.
#' @param learner_timeout (`integer(1)`) \cr
#' Budget (in seconds) for a single parameter evaluation during model training. \cr
#' If this budget is exceeded, the evaluation is stopped and performance measured with the fallback
#' [LearnerClassifFeatureless][mlr3::LearnerClassifFeatureless] or [LearnerRegrFeatureless][mlr3::LearnerRegrFeatureless]. \cr
#' When this is `NULL` (default), the learner timeout defaults to `runtime / 5`.
#' @param resampling ([Resampling][mlr3::Resampling]) \cr
#' Contains the resampling method to be used for hyper-parameter optimization.
#' Defaults to [ResamplingHoldout][mlr3::ResamplingHoldout].
#' @param measure ([Measure][mlr3::Measure]) \cr
#' Contains the performance measure, for which we optimize during training. \cr
#' Defaults to [Accuracy][mlr3measures::acc] for classification and [RMSE][mlr3measures::rmse] for regression.
#' @param runtime (`integer(1)`) \cr
#' Number of seconds for which to run the optimization. Does *not* include training time of the final model. \cr
#' Defaults to `Inf`, letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @param terminator ([Terminator][bbotk::Terminator]) \cr
#' Contains an optional additional termination criterion for model tuning. \cr
#' Note that the [Hyperband][mlr3hyperband] tuner might stop training before the budget is exhausted.
#' [TerminatorRunTime][bbotk::TerminatorRunTime] should not be used, use the separate `runtime` parameter instead. \cr
#' Defaults to [TerminatorNone][bbotk::TerminatorNone], letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @param preprocessing (`character(1)` | [Graph][mlr3pipelines::Graph]) \cr
#' Type of preprocessing to be used. Possible values are :
#' - "none": No preprocessing at all
#' - "stability": [`pipeline_robustify`][mlr3pipelines::pipeline_robustify] is used to guarantee stability of the learners in the pipeline
#' - "full": Adds additional preprocessing operators for [Imputation][mlr3pipelines::PipeOpImpute], [Impact Encoding][mlr3pipelines::PipeOpEncodeImpact] and [PCA][mlr3pipelines::PipeOpPCA]. \cr
#' The choice of preprocessing operators is optimised during tuning.
#'
#' Alternatively, a [Graph][mlr3pipelines::Graph] object can be used to specify a custom preprocessing pipeline.
#' @param portfolio (`logical(1)`) \cr
#' `mlr3automl` tries out a fixed portfolio of known good learners prior to tuning. \cr
#' The `portfolio` parameter disables trying these portfolio learners.
#' @param additional_params ([ParamSet][paradox::ParamSet]) \cr
#' Additional parameter space to tune over, e.g. for custom learners / preprocessing. \cr
#' @param custom_trafo (`function(x, param_set)`) \cr
#' [Trafo function](https://mlr3book.mlr-org.com/searchspace.html#searchspace-trafo)
#' to be applied in addition to existing transformations. Can be used to transform
#' additional_params. \cr
#' @return ([AutoMLClassif][mlr3automl::AutoMLClassif] | [AutoMLRegr][mlr3automl::AutoMLRegr]) \cr
#' Returned class depends on the type of task.
#' @export
#'
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3automl)
#'
#' model = AutoML(tsk("iris"))
#' model$train()
#' }
AutoML = function(task, learner_list = NULL, learner_timeout = NULL,
                  resampling = NULL, measure = NULL, runtime = Inf,
                  terminator = NULL, preprocessing = NULL,
                  portfolio = TRUE, additional_params = NULL,
                  custom_trafo = NULL) {
  if (task$task_type == "classif") {
    # stratify target variable so that every target label appears
    # in all folds while resampling
    target_is_factor = task$col_info[task$col_info$id == task$target_names, ]$type == "factor"
    if (length(target_is_factor) == 1 && target_is_factor) {
      task$col_roles$stratum = task$target_names
    }
    return(AutoMLClassif$new(task, learner_list, learner_timeout,
                             resampling, measure, runtime, terminator,
                             preprocessing, portfolio, additional_params,
                             custom_trafo))
  } else if (task$task_type == "regr") {
    return(AutoMLRegr$new(task, learner_list, learner_timeout,
                          resampling, measure, runtime, terminator,
                          preprocessing, portfolio, additional_params,
                          custom_trafo))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
