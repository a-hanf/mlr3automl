AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, runtime = Inf, terminator = NULL,
                          preprocessing = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      self$measure = measure %??% mlr_measures$get("classif.acc")
      # exclude cv_glmnet and svm by default, because they are slow
      default_learners =  c("classif.ranger", "classif.xgboost", "classif.liblinear")
      self$learner_list = c(learner_list %??% default_learners, "classif.featureless")
      super$initialize(task = task, learner_list = self$learner_list,
                       learner_timeout = learner_timeout, resampling = resampling,
                       measure = self$measure, runtime = runtime, terminator = terminator,
                       preprocessing = preprocessing)
    }
  )
)
