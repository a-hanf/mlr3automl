AutoMLClassif = R6Class(
  "AutoMLClassif",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL,
                          encapsulate = FALSE) {
      checkmate::assert_r6(task, "TaskClassif")
      super$initialize(task, learner_list, resampling, measures,
                       param_set, terminator, encapsulate)
      self$measures = measures %??% mlr_measures$get("classif.acc")
      model$learner_list = learner_list %??% c('classif.ranger')
      self$param_set = private$.get_default_param_set(model$learner_list)
      self$learner = private$.get_default_learner(model$learner_list)
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
      plot(pipeline)
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
      ps = ParamSet$new(list(
        ParamFct$new("branch.selection", learner_list),
        ParamFct$new("classif.ranger.splitrule", c("gini", "extratrees"), tags = "classif.ranger")
      ))
      ps$add_dep(
        "classif.ranger.splitrule", "branch.selection", CondEqual$new("classif.ranger"))
      return(ps)
    }
  )
)
