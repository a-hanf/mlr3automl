AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL,
                          encapsulate = FALSE){
      checkmate::assert_r6(task, "TaskClassif")
      super$initialize(task, learner, resampling, measures,
                       param_set, terminator, encapsulate)
      self$measures = measures %??% mlr_measures$get("classif.acc")
      if (is.null(learner)) {
        self$param_set = private$.get_default_param_set('classif.ranger')
        self$learner = private$.get_default_learner('classif.ranger')
      } else {
        self$param_set = private$.get_default_param_set(learner)
        self$learner = private$.get_default_learner(learner)
      }
    }
  ),
  private = list(
    .get_default_learner = function(learner_list) {
      learners = list()
      for (learner in learner_list) {
        learners = append(learners, private$.create_robust_learner(learner))
      }
      names(learners) = learner_list
      pipeline = ppl("branch", graphs = learners)
      graph_learner = GraphLearner$new(pipeline, task_type = "classif", predict_type = "prob")
      if (self$encapsulate) {
        graph_learner$encapsulate = c(train = "evaluate", predict = "evaluate")
        graph_learner$fallback = lrn("classif.featureless")
      }
      return(AutoTuner$new(graph_learner, self$resampling, self$measures,
                           self$param_set, self$tuning_terminator, self$tuner))
    },
    .create_robust_learner = function(learner_name) {
      pipeline = pipeline_robustify(
        task = self$task,
        learner = lrn(learner_name)
      )
      # subsampling rate is a fidelity parameter in Hyperband (not working yet)
      # pipeline = pipeline %>>% po("subsample")
      pipeline$set_names(pipeline$ids(),
                         paste(learner_name, pipeline$ids(), sep = "."))
      return(pipeline %>>% po("learner", lrn(learner_name, predict_type = "prob")))
    },
    .get_default_param_set = function(learner_list) {
      # TODO: create parameter space dynamically instead of hardcoding
      # Subsampling is only needed for Hyperband
      ps = ParamSet$new(list(
        ParamFct$new("branch.selection", learner_list),
        ParamInt$new("classif.ranger.mtry", lower = 1,
                     upper = length(self$task$feature_names), tags = "classif.ranger"),
        ParamFct$new("classif.ranger.splitrule", c("gini", "extratrees"), tags = "classif.ranger")
      ))
      ps$add_dep(
        "classif.ranger.mtry", "branch.selection", CondEqual$new("classif.ranger"))
      ps$add_dep(
        "classif.ranger.splitrule", "branch.selection", CondEqual$new("classif.ranger"))
      return(ps)
    }
  )
)
