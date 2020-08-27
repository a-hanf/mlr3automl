AutoMLRegr = R6Class(
  "AutoMLRegr",
  inherit = AutoMLBase,
  public = list(
    initialize = function(task, learner_list = NULL, resampling = NULL,
                          measures = NULL, param_set = NULL, terminator = NULL,
                          encapsulate = FALSE) {
      checkmate::assert_r6(task, "TaskRegr")
      super$initialize(task, learner_list, resampling, measures,
                       param_set, terminator, encapsulate)
      self$measures = measures %??% mlr_measures$get("regr.mae")
      model$learner_list = learner_list %??% c('regr.ranger')
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
      pipeline$set_names(pipeline$ids(),
                         paste(learner_name, pipeline$ids(), sep = "."))
      if (learner_name == 'regr.ranger') private$.set_mtry_for_random_forest(pipeline)

      pipeline = pipeline %>>% po("learner", lrn(learner_name, predict_type = "prob"))
      return(pipeline)
    },
    .set_mtry_for_random_forest = function(pipeline) {
      pipe_copy = pipeline$clone(deep = TRUE) %>>% lrn('regr.ranger', num.trees = 1)
      pipe_copy$train(self$task)
      num_effective_vars = length(pipe_copy$state$classif.ranger$train_task$feature_names)
      self$param_set$add(
        ParamDbl$new("regr.ranger.mtry", lower = floor(num_effective_vars^0.1),
                     upper = floor(num_effective_vars^0.9), tags = "regr.ranger"))
      self$param_set$add_dep(
        "regr.ranger.mtry", "branch.selection", CondEqual$new("regr.ranger"))
    },
    .get_default_param_set = function(learner_list) {
      # TODO: create parameter space dynamically instead of hardcoding
      # Subsampling is only needed for Hyperband
      ps = ParamSet$new(list(
        ParamFct$new("branch.selection", learner_list),
        ParamInt$new("regr.ranger.mtry", lower = 1,
                     upper = length(self$task$feature_names), tags = "regr.ranger"),
        ParamFct$new("regr.ranger.splitrule", c("variance", "extratrees"), tags = "regr.ranger")
      ))
      ps$add_dep(
        "regr.ranger.mtry", "branch.selection", CondEqual$new("regr.ranger"))
      ps$add_dep(
        "regr.ranger.splitrule", "branch.selection", CondEqual$new("regr.ranger"))
      return(ps)
    }
  )
)
