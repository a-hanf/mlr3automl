#' @title AutoMLRegr
#'
#' @description
#' Class for Automated Regression in mlr3automl. Subclass of [AutoMLBase][mlr3automl::AutoMLBase]
#'
#' @section Construction:
#' Objects should be created using the [AutoML][mlr3automl::AutoML] interface function.
#' ```
#' regression_model = AutoML(regression_task, learner_list, learner_timeout, resampling, measure, runtime,
#'                           terminator, preprocessing, portfolio)
#' ```
#'
#' @param task ([`Task`][mlr3::Task]) \cr
#' [`TaskRegr`][mlr3::TaskRegr] to be solved.
#' @param learner_list (`list()` | `character()`) \cr
#' `List` of names from [mlr_learners][mlr3::mlr_learners]. Can be used to customize the learners to be tuned over. \cr
#' Default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")` \cr
#' Might break mlr3automl if a user-provided learner is incompatible with the provided task.
#' @param learner_timeout (`integer(1)`) \cr
#' Budget (in seconds) for a single parameter evaluation during model training. \cr
#' If this budget is exceeded, the evaluation is stopped and performance measured with the fallback
#' [LearnerRegrFeatureless][mlr3::LearnerRegrFeatureless]. \cr
#' When this is `NULL` (default), the learner timeout defaults to `runtime / 5`.
#' @param resampling ([Resampling][mlr3::Resampling]) \cr
#' Contains the resampling method to be used for hyper-parameter optimization.
#' Defaults to [ResamplingHoldout][mlr3::ResamplingHoldout].
#' @param measure ([Measure][mlr3::Measure]) \cr
#' Contains the performance measure, for which we optimize during training. \cr
#' Defaults to [RMSE][mlr3measures::acc].
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
#'
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3automl)
#'
#' regr_model = AutoML(tsk("mtcars"))
#' regr_model$train()
#' }
AutoMLRegr = R6Class(
  "AutoMLRegr",
  inherit = AutoMLBase,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #'
    #' @return [AutoMLRegr][mlr3automl::AutoMLRegr]
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, runtime = Inf, terminator = NULL,
                          preprocessing = NULL, portfolio = TRUE){
      checkmate::assert_r6(task, "TaskRegr")
      self$measure = measure %??% mlr_measures$get("regr.rmse")
      default_learners =  c("regr.ranger", "regr.xgboost","regr.svm",
                            "regr.liblinear", "regr.cv_glmnet")
      self$learner_list = learner_list %??% default_learners
      if (!("regr.featureless" %in% self$learner_list)) {
        self$learner_list = c(self$learner_list, "regr.featureless")
      }
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measure = self$measure, runtime = runtime, terminator = terminator,
                       preprocessing = preprocessing, portfolio = portfolio)
    }
  )
)
