AutoMLRegr = R6Class(
  "AutoMLRegr",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, resampling = NULL,
                          measures = NULL, terminator = NULL){
      checkmate::assert_r6(task, "TaskRegr")
      self$measures = measures %??% mlr_measures$get("regr.mae")
      self$learner_list = learner_list %??% c('regr.ranger')
      super$initialize(task, learner_list, resampling, measures, terminator)
    }
  )
)
