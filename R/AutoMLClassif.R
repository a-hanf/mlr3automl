AutoMLClassif <- R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL) {
      checkmate::assert_r6(task, "TaskClassif")
      super$initialize(task, learner, resampling, measures, param_set, terminator)
      self$param_set <- if (!is.null(param_set)) param_set else private$..get_default_param_set()
      self$measures <- if (!is.null(measures)) measures else mlr_measures$get("classif.acc")
      self$learner <- if (!is.null(learner)) learner else private$..get_default_learner()
    }
  ),
  private = list(
    ..get_default_learner = function() {
      pipeline <- ppl("branch", graphs = list(
        decision_tree = private$..create_robust_learner("classif.rpart"),
        random_forest = private$..create_robust_learner("classif.ranger"),
        xgboost = private$..create_robust_learner("classif.xgboost"),
        # svm has some weird issues with factors
        # svm = private$..create_robust_learner("classif.svm"),
        baseline = private$..create_robust_learner("classif.log_reg")
      ))
      plot(pipeline)
      return(AutoTuner$new(GraphLearner$new(pipeline, task_type = "classif"), self$resampling, self$measures, self$param_set, self$tuning_terminator, self$tuner))
    },
    ..create_robust_learner = function(learner_name) {
      pipeline <- pipeline_robustify(
        task = self$task,
        learner = lrn(learner_name)
      )
      # subsampling is used for the fidelity parameter in Hyperband, which isn't working yet
      # pipeline = pipeline %>>% po("subsample")
      pipeline$set_names(pipeline$ids(), paste(learner_name, pipeline$ids(), sep = "."))
      return(pipeline %>>% po("learner", lrn(learner_name)))
    },
    ..get_default_param_set = function() {
      # TODO: create parameter space dynamically instead of hardcoding
      ps = ParamSet$new(list(
        ParamFct$new("branch.selection", c("decision_tree", "random_forest", "xgboost", "baseline")),
        # Subsampling is only needed for Hyperband
        # ParamDbl$new("classif.rpart.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "decision_tree")),
        ParamInt$new("classif.rpart.minsplit", lower = 1, upper = dim(task$data())[1], tags = "decision_tree"),
        ParamDbl$new("classif.rpart.cp", lower = 0, upper = 1, tags = "decision_tree"),

        # ParamDbl$new("classif.ranger.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "random_forest")),
        ParamInt$new("classif.ranger.mtry", lower = 1, upper = length(task$feature_names), tags = "random_forest"),

        # ParamDbl$new("classif.xgboost.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "xgboost")),
        ParamInt$new("classif.xgboost.nrounds", lower = 1, upper = 200, tags = "xgboost"),
        ParamDbl$new("classif.xgboost.eta", lower = 0, upper = 1, tags = "xgboost"),
        ParamDbl$new("classif.xgboost.gamma", lower = 0, upper = 5, tags = "xgboost")

        # ParamDbl$new("classif.log_reg.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "baseline"))
      ))
      # ps$add_dep("classif.rpart.subsample.frac", "branch.selection", CondEqual$new("decision_tree"))
      ps$add_dep("classif.rpart.minsplit", "branch.selection", CondEqual$new("decision_tree"))
      ps$add_dep("classif.rpart.cp", "branch.selection", CondEqual$new("decision_tree"))

      # ps$add_dep("classif.ranger.subsample.frac", "branch.selection", CondEqual$new("random_forest"))
      ps$add_dep("classif.ranger.mtry", "branch.selection", CondEqual$new("random_forest"))

      # ps$add_dep("classif.xgboost.subsample.frac", "branch.selection", CondEqual$new("xgboost"))
      ps$add_dep("classif.xgboost.nrounds", "branch.selection", CondEqual$new("xgboost"))
      ps$add_dep("classif.xgboost.eta", "branch.selection", CondEqual$new("xgboost"))
      ps$add_dep("classif.xgboost.gamma", "branch.selection", CondEqual$new("xgboost"))

      # ps$add_dep("classif.log_reg.subsample.frac", "branch.selection", CondEqual$new("baseline"))
      return(ps)
    }
  )
)
