AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      self$measure = measure %??% mlr_measures$get("classif.acc")
      # exclude cv_glmnet and svm by default, because they are slow
      self$learner_list = learner_list %??% c(
        "classif.ranger", "classif.xgboost",
        "classif.liblinear")
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measure = self$measure, terminator = terminator)
    }
  )
)
