AutoMLRegr = R6Class(
  "AutoMLRegr",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measures = NULL, terminator = NULL){
      checkmate::assert_r6(task, "TaskRegr")
      self$measures = measures %??% mlr_measures$get("regr.rmse")
      self$learner_list = learner_list %??% c(
        "regr.ranger", "regr.xgboost",
        "regr.svm", "regr.liblinear",
        "regr.cv_glmnet")
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measures = self$measures, terminator = terminator)
    }
  )
)
