AutoMLRegr = R6Class(
  "AutoMLRegr",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, runtime = Inf, terminator = NULL,
                          preprocessing = NULL, portfolio = TRUE){
      checkmate::assert_r6(task, "TaskRegr")
      self$measure = measure %??% mlr_measures$get("regr.rmse")
      default_learners =  c("regr.ranger", "regr.xgboost","regr.svm",
                            "regr.liblinear", "regr.cv_glmnet")
      self$learner_list = c(learner_list %??% default_learners, "regr.featureless")
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measure = self$measure, runtime = runtime, terminator = terminator,
                       preprocessing = preprocessing, portfolio = portfolio)
    }
  )
)
