AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, resampling = NULL,
                          measures = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      self$measures = measures %??% mlr_measures$get("classif.acc")
      self$learner_list = learner_list %??% c("classif.ranger", "classif.xgboost")
      super$initialize(task, learner_list, resampling, measures, terminator)
    }
  )
)
