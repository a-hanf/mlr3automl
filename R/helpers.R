#' @title Create an AutoTuner with a single line of code
#'
#' @description
#' Small utility function, which creates an [AutoTuner][mlr3tuning::AutoTuner] for a given [learner][mlr3::Learner]. \cr
#' Uses the same interface as the [AutoTuner][mlr3tuning::AutoTuner], but provides defaults for all arguments. \cr
#' Parameter spaces are identical to the ones used in [mlr3automl].
#' @param learner (`character(1)` | [Learner][mlr3::Learner]) \cr
#' Learner inside the [AutoTuner][mlr3tuning::AutoTuner]. Parameter sets are predefined for
#' `ranger`, `xgboost`, `liblinear`, `svm` and `cv_glmnet` learners for both
#' prediction and regression. Other learners will obtain empty parameter sets.
#' @param resampling ([Resampling][mlr3::Resampling]) \cr
#' @param measure ([Measure][mlr3::Measure]) \cr
#' @param terminator ([Terminator][bbotk::Terminator])\cr
#' @param tuner ([Tuner][mlr3tuning::Tuner] | [TunerHyperband][mlr3hyperband::TunerHyperband])\cr
#' Tuner. Hyperband is supported by creating a [`GraphLearner`][mlr3pipelines::GraphLearner]
#' with [`PipeOpSubsample`][mlr3pipelines::PipeOpSubsample].
#' @param num_effective_vars \cr
#' Integer giving the number of features in the dataset. Only required for
#' parameter transformation of `mtry` in Random Forest (we are tuning over
#' `num_effective_vars^0.1` to `num_effective_vars^0.9`)
#' @return [AutoTuner][mlr3tuning::AutoTuner]
#' @examples
#' \donttest{
#' library(mlr3automl)
#' my_autotuner = create_autotuner(lrn("classif.svm"))
#' }
#' @export
create_autotuner = function(
  learner = lrn("classif.xgboost"), resampling = rsmp("cv", folds = 10),
  measure = NULL, terminator = trm("run_time", secs = 60), tuner = tnr("random_search"),
  num_effective_vars = NULL) {

  if (is.character(learner)) {
    assert_character(learner, any.missing = FALSE, len = 1)
    learner = lrn(learner)
  }

  assert_learner(learner)
  assert_resampling(resampling)
  if (!is.null(measure)) assert_measure(measure)
  assert_int(num_effective_vars, null.ok = TRUE)

  task_type = sub("\\..*", "", learner$id)
  default_msr = get(task_type, mlr_reflections$default_measures)

  if (grepl("ranger", learner$id) && is.null(num_effective_vars)) {
    warning("For tuning of Random Forest, the number of features in the dataset
            should be provided. Defaulting to 10")
    num_effective_vars = 10
  }

  if ("TunerHyperband" %in% class(tuner)) {
    learner = GraphLearner$new(po("subsample") %>>% learner, id = learner$id)
    using_hyperband = TRUE
  } else {
    using_hyperband = FALSE
  }

  params = default_params(learner_list = learner$id,
                          feature_counts = matrix(num_effective_vars, ncol = 2, nrow = 3, byrow = TRUE, dimnames = list(c("no_encoding", "one_hot_encoding", "impact_encoding"), c("numeric_cols", "all_cols"))),
                          using_hyperband,
                          using_prefixes = using_hyperband,
                          preprocessing = "none")

  return(AutoTuner$new(
    learner = learner,
    resampling = resampling,
    measure = measure %??% msr(default_msr),
    search_space = params,
    terminator = terminator,
    tuner = tuner))
}

remove_existing_edges = function(current_pipeline, existing_pipeop) {
  # remove source and destination edge of existing node
  current_pipeline$edges = current_pipeline$edges[!(current_pipeline$edges$src_id == existing_pipeop | current_pipeline$edges$dst_id == existing_pipeop), ]
}

get_predecessor_successor = function(current_pipeline, existing_pipeop) {
  predecessor = current_pipeline$edges[current_pipeline$edges$dst_id == existing_pipeop, ]$src_id
  successor = current_pipeline$edges[current_pipeline$edges$src_id == existing_pipeop, ]$dst_id
  return(c(predecessor = predecessor, successor = successor))
}

add_branching = function(current_pipeline, choices, id, columns) {
  # branch over imputation methods for numerical columns
  current_pipeline$add_pipeop(po("branch", options = choices, id = id))

  # add new pipeops and edges
  for (pipeop in choices) {
    if (!(pipeop %in% current_pipeline$ids())) {
      pipeop_name = sub(".*\\.", "", pipeop)
      if (pipeop_name != "nop") {
        current_pipeline$add_pipeop(po(pipeop_name, affect_columns = selector_type(columns), id = pipeop))
      } else {
        current_pipeline$add_pipeop(po(pipeop_name, id = pipeop))
      }

    }
    current_pipeline$add_edge(id, pipeop, src_channel = pipeop)
  }
}

add_unbranching = function(current_pipeline, choices, id) {
  current_pipeline$add_pipeop(po("unbranch", options = length(choices), id = id))

  for (pipeop_idx in seq_along(choices)) {
    current_pipeline$add_edge(src_id = choices[pipeop_idx], dst_id = id, dst_channel = pipeop_idx)
  }
}

replace_existing_node = function(current_pipeline, existing_pipeop, pipeop_choices, branching_prefix, columns) {
  # get predecessor and successor of node to be replaced
  neighbor_nodes = get_predecessor_successor(current_pipeline, existing_pipeop)

  # remove source and destination edge of existing node
  remove_existing_edges(current_pipeline, existing_pipeop)

  # add new branching
  add_branching(current_pipeline,
                choices = pipeop_choices,
                id = paste0(branching_prefix, "branch"),
                columns = columns)

  # add unbranching pipeop and connect
  add_unbranching(current_pipeline,
                  choices = pipeop_choices,
                  id = paste0(branching_prefix, "unbranch"))

  # connect new subgraph to predecessor and successor
  if (!is.na(neighbor_nodes["successor"])) {
    current_pipeline$add_edge(paste0(branching_prefix, "unbranch"), neighbor_nodes["successor"])
  }

  if (!is.na(neighbor_nodes["predecessor"])) {
    current_pipeline$add_edge(neighbor_nodes["predecessor"], paste0(branching_prefix, "branch"))
  }
}

