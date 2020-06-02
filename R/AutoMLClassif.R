AutoMLClassif <- R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      super$initialize(task, learner, resampling, measures, param_set, terminator)
      self$param_set <- if (!is.null(param_set)) {
        param_set
      } else {
        ParamSet$new(list(ParamFct$new("branch.selection", c("decision_tree", "xgboost", "svm", "baseline"))))
        # ParamInt$new("pca.rank.", lower = 1, upper = 4)))
        # the pca.rank. parameter is only interesting when the pca branch is taken
        # ps$add_dep("pca.rank.", "branch.selection", CondEqual$new("pca"))
      }
      self$measures <- if (!is.null(measures)) measures else mlr_measures$get("classif.acc")
      self$learner <- if (!is.null(learner)) learner else private$..get_default_learner()
    }
  ),
  private = list(
    ..get_default_learner = function() {
      pipeline <- ppl("branch", graphs = list(
        decision_tree = private$..create_robust_learner("classif.rpart"),
        xgboost = private$..create_robust_learner("classif.xgboost"),
        svm = private$..create_robust_learner("classif.svm"),
        baseline = private$..create_robust_learner("classif.log_reg")
      ))
      plot(pipeline)
      return(AutoTuner$new(GraphLearner$new(pipeline), self$resampling, self$measures, self$param_set, self$tuning_terminator, self$tuner))
    },
    ..create_robust_learner = function(learner_name) {
      pipeline <- pipeline_robustify(
        task = self$task,
        learner = lrn(learner_name)
      )
      pipeline$set_names(pipeline$ids(), paste(learner_name, pipeline$ids(), sep = "."))
      return(pipeline %>>% lrn(learner_name))
    }
  )
)
