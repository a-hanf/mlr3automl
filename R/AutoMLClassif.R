AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      super$initialize(task, learner, resampling, measures, param_set, terminator)
      self$param_set = if (!is.null(param_set)) {
        param_set
      } else {
        ParamSet$new(list(
          ParamDbl$new("classif.rpart.cp", lower = 0.001, upper = 0.1),
          ParamInt$new("classif.rpart.minsplit", lower = 1, upper = 10)
        ))
      }
      self$measures = if (!is.null(measures)) measures else mlr_measures$get("classif.acc")
      self$learner <- if (!is.null(learner)) learner else private$..get_default_learner()
    }
  ),
  private = list(
    ..get_default_learner = function() {
      pipeline = po("imputemedian") %>>%
        po("learner", learner = mlr_learners$get("classif.rpart"))
      return(AutoTuner$new(GraphLearner$new(pipeline), self$resampling, self$measures, self$param_set, self$tuning_terminator, self$tuner))
    }
  )
)
