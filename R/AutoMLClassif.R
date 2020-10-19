AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measures = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      self$measures = measures %??% mlr_measures$get("classif.acc")
      self$learner_list = learner_list %??% c(
        "classif.ranger", "classif.xgboost",
        "classif.svm", "classif.liblinearl1l2svc",
        "classif.liblinearl1logreg", "classif.cv_glmnet")
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measures = self$measures, terminator = terminator)
    }
  )
)
