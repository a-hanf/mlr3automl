AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measures = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      self$measures = measures %??% mlr_measures$get("classif.acc")
      # exclude cv_glmnet and svm by default, because they are slow
      self$learner_list = learner_list %??% c(
        "classif.ranger", "classif.xgboost",
        "classif.liblinear")
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measures = self$measures, terminator = terminator)
    }
  )
)
