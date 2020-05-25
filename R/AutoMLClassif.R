library(R6)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3)
library(paradox)
library(checkmate)

AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoML,
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
    }
  ),
  private = list(
    ..get_default_learner = function() {
      pipeline = po("imputemedian") %>>%
        po("learner", learner = mlr_learners$get("classif.rpart"))
      return(GraphLearner$new(pipeline))
    }
  )
)
