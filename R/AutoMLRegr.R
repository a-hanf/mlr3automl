AutoMLRegr = R6Class(
  "AutoMLRegr",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL){
      checkmate::assert_r6(task, "TaskRegr")
      super$initialize(task, learner, resampling, measures,
                       param_set, terminator)
      self$param_set = param_set %??% private$.get_default_param_set()
      self$measures = measures %??% mlr_measures$get("regr.mae")
      self$learner = learner %??% private$.get_default_learner()
    }
  ),
  private = list(
    .get_default_learner = function() {
      pipeline = ppl("branch", graphs = list(
        decision_tree = private$.create_robust_learner("regr.rpart"),
        random_forest = private$.create_robust_learner("regr.ranger"),
        xgboost = private$.create_robust_learner("regr.xgboost")
      ))
      plot(pipeline)
      graph_learner = GraphLearner$new(pipeline, task_type = "regr")
      if (self$encapsulate) {
        graph_learner$encapsulate = c(train = "evaluate", predict = "evaluate")
        graph_learner$fallback = lrn("regr.featureless")
      }
      return(AutoTuner$new(graph_learner, self$resampling, self$measures,
                           self$param_set, self$tuning_terminator, self$tuner))
    },
    .create_robust_learner = function(learner_name) {
      pipeline = pipeline_robustify(
        task = self$task,
        learner = lrn(learner_name)
      )
      # subsampling is used for the fidelity parameter in Hyperband, which isn't working yet
      # pipeline = pipeline %>>% po("subsample")
      pipeline$set_names(pipeline$ids(),
                         paste(learner_name, pipeline$ids(), sep = "."))
      return(pipeline %>>% po("learner", lrn(learner_name)))
    },
    .get_default_param_set = function() {
      # TODO: create parameter space dynamically instead of hardcoding
      ps = ParamSet$new(list(
        ParamFct$new("branch.selection",
                     c("decision_tree", "random_forest", "xgboost")),
        # Subsampling is only needed for Hyperband
        # ParamDbl$new("regr.rpart.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "decision_tree")),
        ParamInt$new("regr.rpart.minsplit", lower = 1,
                     upper = dim(self$task$data())[1], tags = "decision_tree"),
        ParamDbl$new(
          "regr.rpart.cp", lower = 0, upper = 1, tags = "decision_tree"),

        # ParamDbl$new("regr.ranger.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "random_forest")),
        ParamInt$new("regr.ranger.mtry", lower = 1,
                     upper = length(self$task$feature_names), tags = "random_forest"),

        # ParamDbl$new("regr.xgboost.subsample.frac", lower = 0.1, upper = 1, tags = c("budget", "xgboost")),
        ParamInt$new(
          "regr.xgboost.nrounds", lower = 1, upper = 200, tags = "xgboost"),
        ParamDbl$new(
          "regr.xgboost.eta", lower = 0, upper = 1, tags = "xgboost"),
        ParamDbl$new(
          "regr.xgboost.gamma", lower = 0, upper = 5, tags = "xgboost")
      ))
      # ps$add_dep("regr.rpart.subsample.frac", "branch.selection", CondEqual$new("decision_tree"))
      ps$add_dep(
        "regr.rpart.minsplit", "branch.selection", CondEqual$new("decision_tree"))
      ps$add_dep(
        "regr.rpart.cp", "branch.selection", CondEqual$new("decision_tree"))

      # ps$add_dep("regr.ranger.subsample.frac", "branch.selection", CondEqual$new("random_forest"))
      ps$add_dep(
        "regr.ranger.mtry", "branch.selection", CondEqual$new("random_forest"))

      # ps$add_dep("regr.xgboost.subsample.frac", "branch.selection", CondEqual$new("xgboost"))
      ps$add_dep(
        "regr.xgboost.nrounds", "branch.selection", CondEqual$new("xgboost"))
      ps$add_dep(
        "regr.xgboost.eta", "branch.selection", CondEqual$new("xgboost"))
      ps$add_dep(
        "regr.xgboost.gamma", "branch.selection", CondEqual$new("xgboost"))
      return(ps)
    }
  )
)
