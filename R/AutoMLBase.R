#' @title AutoML
#' @format [R6Class] AutoML
#' @usage NULL
#' @format [`R6Class`].
#' @description
#' base class for AutoML in mlr3automl. Has subclasses for Classification and Regression.
#' @section Construction:
#' ```
#' AutoMLBase$new(task)
#' ```
#' @section Internals:
#' The AutoML class uses `mlr3pipelines` to create a machine learning pipeline.
#' This pipeline contains multiple models (decision tree, random forest, XGBoost),
#' which are wrapped in a GraphLearner. This GraphLearner is wrapped in an
#' AutoTuner for Hyperparameter Optimization and during training or resampling.
#' @section Fields:
#' * `task` :: `Task` object from `mlr3` \cr
#'   Contains the data and some meta-features (like the target variable)
#' * `learner_list` :: `List` of names for `mlr3 Learners` \cr
#'   Can be used to customize the learners to be tuned over. If no parameter space
#'   is defined for the selected learner, it will be run with default parameters.
#'   Might break mlr3automl if the learner is incompatible with the provided task
#' * `learner_timeout` :: `Integer` \cr
#'   Budget (in seconds) for a single learner during training of the pipeline
#' * `preprocessing` :: `Character` \cr
#'   Type of preprocessing to be used. Possible values are "none", "stability"
#'   and "full". Alternatively, a `mlr3pipelines::Graph` object can be used
#'   to specify a custom preprocessing pipeline.
#' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
#'   Contains the resampling method to be used for hyper-parameter optimization
#' * `measure` :: `Measure` object from `mlr_measures` \cr
#'   Contains the performance measure, for which we optimize during training
#' * `tuning_terminator` :: `Terminator` object from `bbotk` \cr
#'   Contains the termination criterion for model tuning
#' * `tuner` :: `Tuner` object from `mlr3tuning` \cr
#'   Type of tuning. We use Hyperband.
#' @section Methods:
#' * `train()` \cr
#'   Trains the AutoML system.
#' * `predict(data = NULL, row_ids = NULL)` \cr
#'   `data.frame | data.table | Task -> PredictionClassif or PredictionRegr`
#'   Returns a Prediction object for the given data based on the trained model.
#'   If data is NULL, defaults to the task used for training
#'   `resample()`
#'   `double(1) -> ResampleResult`
#'   Performs nested resampling with a train/test split as the outer resampling
#' @rawNamespace import(mlr3, except = c(lrn, lrns))
#' @import mlr3learners
#' @import mlr3extralearners
#' @import mlr3hyperband
#' @import mlr3misc
#' @import mlr3oml
#' @import mlr3pipelines
#' @import mlr3tuning
#' @import paradox
#' @import checkmate
#' @import testthat
#' @import glmnet
#' @import xgboost
#' @importFrom R6 R6Class
#' @name AutoMLBase
#' @examples
#' "add later"
AutoMLBase = R6Class("AutoMLBase",
  public = list(
    task = NULL,
    learner_list = NULL,
    learner_timeout = NULL,
    learner = NULL,
    preprocessing = NULL,
    resampling = NULL,
    measure = NULL,
    tuning_terminator = NULL,
    runtime = NULL,
    tuner = NULL,
    #' @description
    #' Creates a new AutoMLBase object
    #' @param task
    #' * `task` :: `Task` object from `mlr3` \cr
    #'   Contains the task to be solved.
    #' @param learner_list
    #' * `learner_list` :: `List` of names for `mlr3 Learners` \cr
    #'   Can be used to customize the learners to be tuned over. If no parameter space
    #'   is defined for the selected learner, it will be run with default parameters.
    #'   Default learners for classification: `c("classif.ranger", "classif.xgboost", "classif.liblinear")`,
    #'   default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")`.
    #'   Might break mlr3automl if the learner is incompatible with the provided task.
    #' @param learner_timeout
    #' * `learner_timeout` :: `Integer` \cr
    #'   Budget (in seconds) for a single learner during resampling of the pipeline.
    #'   If this budget is exceeded, the learner is replaced with the fallback
    #'   learner (`lrn("classif.featureless")` or `lrn("regr.featureless")`).
    #'   When this is `NULL` (default), the learner timeout is set to `clock_time / 5`.
    #' @param resampling
    #' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
    #'   Contains the resampling method to be used for hyper-parameter optimization.
    #'   Defaults to `rsmp("holdout")`.
    #' @param measure
    #' * `measure` :: `Measure` object from `mlr_measures` \cr
    #'   Contains the performance measure, for which we optimize during training.
    #'   Defaults to `msr("classif.acc")` for classification and `msr("regr.rmse")`
    #'   for regression.
    #' @param runtime
    #'  * `runtime` :: `numeric(1)`\cr
    #'    Number of seconds for which to run the optimization. Does *not* include training time of the final model.
    #' @param terminator
    #' * `terminator` :: `Terminator` object from `mlr3tuning` \cr
    #'   Contains an optional additional termination criterion for model tuning. Note that the Hyperband
    #'   tuner might stop training before the budget is exhausted.
    #'   Note also that no `runtime` terminator needs to be given, as the `runtime` is given separately.
    #'   Defaults to `trm("none")`
    #' @param preprocessing
    #' * `preprocessing` :: `Character` \cr
    #'   Type of preprocessing to be used. Possible values are "none", "stability"
    #'   and "full". Alternatively, a `mlr3pipelines::Graph` object can be used
    #'   to specify a custom preprocessing pipeline.
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, runtime = Inf, terminator = NULL,
                          preprocessing = NULL) {

      assert_task(task)
      assert_character(learner_list, any.missing = FALSE, min.len = 1)
      for (learner in learner_list) {
        assert_subset(learner, mlr_learners$keys())
      }

      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measure)) assert_measure(measure)

      self$task = task
      self$resampling = resampling %??% rsmp("holdout")
      self$preprocessing = preprocessing %??% "stability"

      self$runtime = assert_number(runtime, lower = 0)
      self$learner_timeout = assert_number(learner_timeout, lower = 0, null.ok = TRUE) %??% runtime / 5  # maybe choose a larger divisor here
      self$tuning_terminator = terminator %??% trm("none")

      self$tuner = tnr("hyperband", eta = 3)
      self$learner = private$.get_default_learner()
    },
    #' @description
    #' Train AutoML learner. Calls the `train` method of the associated `AutoTuner`
    #' with the training instances in the given task.
    #' @param row_ids
    #' IDs of observations to be used for training. If no `row_ids` are provided,
    #' trains on the entire data set.
    train = function(row_ids = NULL) {
      self$learner$train(self$task, row_ids)
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
      }
    },
    #' @description
    #' Make predictions for new observations
    #' @param data
    #' Optional. If provided, predictions are made on this dataset. Needs to have
    #' the same format as data used for training.
    #' @param row_ids
    #' IDs of observations to be used for predictions If no `row_ids` are provided,
    #' predictions are made for the entire dataset.
    predict = function(data = NULL, row_ids = NULL) {
      if (is.null(data)) {
        return(self$learner$predict(self$task, row_ids))
      } else {
        return(self$learner$predict(data, row_ids))
      }
    },
    #' @description
    #' Convenience function for resampling with an AutoML Object. Performs nested
    #' resampling with `$resampling` as inner resampling and `rsmp("holdout")`
    #' as outer resampling.
    resample = function() {
      outer_resampling = rsmp("holdout")
      resample_result = mlr3::resample(self$task, self$learner, outer_resampling)
      self$learner = resample_result$learners[[1]]
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
      }
      return(resample_result)
    },
    #' @description
    #' Convenience function for trained AutoML objects. Extracts the best
    #' performing hyperparameters.
    tuned_params = function() {
      if (is.null(self$learner$tuning_instance$archive)) {
        warning("Model has not been trained. Run the $train() method first.")
      } else {
        return(self$learner$tuning_instance$archive$best())
      }
    }
  ),
  private = list(
    .get_default_learner = function() {
      # number of variables is needed for setting mtry in ranger
      # also for setting max rank in PCA/ICA during feature preprocessing
      if (any(grepl("ranger", self$learner_list)) || self$preprocessing == "full") {
        feature_counts = private$.compute_num_effective_vars()
      } else {
        feature_counts = NULL
      }

      preprocessing_pipeops = private$.get_preprocessing_pipeline(min(feature_counts[, "numeric_cols"]))

      learners = lapply(self$learner_list, function(x) private$.create_robust_learner(x))
      names(learners) = self$learner_list
      if (self$task$task_type == "classif") {
        pipeline = preprocessing_pipeops %>>% po("subsample", stratify = TRUE)
      } else {
        pipeline = preprocessing_pipeops %>>% po("subsample")
      }

      if (length(self$learner_list) > 1) {
        pipeline =  pipeline %>>% ppl("branch", graphs = learners)
      } else {
        pipeline = pipeline %>>% learners[[1]]
      }
      graph_learner = GraphLearner$new(pipeline, id = "mlr3automl_pipeline")

      # # fallback learner is featureless learner for classification / regression
      # graph_learner$fallback = lrn(paste(self$task$task_type, '.featureless',
      #                                    sep = ""))
      # # use callr encapsulation so we are able to kill model training, if it
      # # takes too long
      # graph_learner$encapsulate = c(train = "callr", predict = "callr")
      # graph_learner$timeout = c(train = self$learner_timeout,
      #                           predict = self$learner_timeout)

      param_set = default_params(learner_list = self$learner_list,
                                 feature_counts = feature_counts,
                                 preprocessing = self$preprocessing,
                                 feature_types = unique(self$task$feature_types$type))

      tuner = self$tuner
      initial_design = get_portfolio_design(self$task$task_type, param_set, self$learner_list)
      if (nrow(initial_design) > 0) {
        tuner_list = list(tnr("design_points", design = initial_design), self$tuner)
      } else {
        tuner_list = list(self$tuner)
      }

      tuner = TunerChain$new(tuner_list)

      if (is.finite(self$runtime)) {
        tuner = TunerWrapperHardTimeout$new(
          tuner,
          timeout = self$runtime
        )
      }

      return(AutoTuner$new(
        learner = graph_learner,
        resampling = self$resampling,
        measure = self$measure,
        search_space = param_set,
        terminator = self$tuning_terminator,
        tuner = tuner
      ))
    },
    .get_preprocessing_pipeline = function(ncol_numeric = NULL) {
      if (any(grepl("Graph|PipeOp", class(self$preprocessing)))) {
        return(self$preprocessing)
      } else  if (self$preprocessing == "none") {
        return(NULL)
      }

      # po("nop") is needed so we have a predecessor for the imputation nodes
      stability_preprocessing = po("nop", id = "start") %>>% pipeline_robustify(self$task, impute_missings = TRUE, factors_to_numeric = FALSE)
      if (any(c("factor", "ordered", "character") %in% self$task$col_info$type)) {
        stability_preprocessing = stability_preprocessing %>>% po("encodeimpact")
      }
      stability_preprocessing$update_ids(prefix = "stability.")

      # renaming is needed, otherwise we have two PipeOps called
      # "imputesample" (in factor imputation and later on for stability of
      # the fixfactors operator)
      if ("stability.imputeoor" %in% stability_preprocessing$ids()) {
        stability_preprocessing$set_names("stability.imputeoor", "imputation.imputeoor")
      }
      if ("stability.imputehist" %in% stability_preprocessing$ids()) {
        stability_preprocessing$set_names("stability.imputehist", "imputation.imputehist")
      }

      if (self$preprocessing == "stability") {
        return(stability_preprocessing)
      }

      # for feature preprocessing,  we add more imputation / encoding methods
      # as well as dimensionality reduction pipeops and tune over their params

      # first, add more imputation / encoding ops to existing pipeline
      private$.extend_preprocessing(stability_preprocessing)

      dimensionality_reduction = list(po("pca"), po("nop"))
      names(dimensionality_reduction) = sapply(dimensionality_reduction, function(x) paste0("dimensionality.", x$id))
      return(stability_preprocessing %>>% po("scale") %>>% ppl("branch", graphs = dimensionality_reduction)$update_ids(prefix = "dimensionality."))
    },
    .create_robust_learner = function(learner_name) {
      # liblinear only works with columns of type double. Convert ints / bools -> dbl
      if (grepl('liblinear', learner_name)) {
        pipeline = po("colapply", applicator = as.numeric,
             param_vals = list(affect_columns = selector_type(c("logical", "integer"))))
      } else {
        pipeline = NULL
      }

      # predict probabilities for classification if possible
      if (self$task$task_type == "classif" && ("prob" %in% lrn(learner_name)$predict_types)) {
        return(pipeline %>>% po("learner", lrn(learner_name, predict_type = "prob")))
      }
      # default: predict with type response
      return(pipeline %>>% po("learner", lrn(learner_name)))
    },
    .compute_num_effective_vars = function() {
      base_pipeline =
        private$.get_preprocessing_pipeline() %>>%
        lrn(paste(self$task$task_type, '.featureless', sep = ""))

      result = matrix(nrow = 0, ncol = 2, byrow = TRUE)
      colnames(result) = c("numeric_cols", "all_cols")

      # get number of variables after preprocessing
      last_pipeop = paste(self$task$task_type, '.featureless', sep = "")

      # dummy param is needed here to get an evaluation with the tuner
      preprocessing_params = list(impact_encoding = ParamSet$new(list(
        ParamDbl$new("stability.removeconstants.ratio", lower = 1e-4, upper = 1e-4))))

      if (self$preprocessing == "full") {
        preprocessing_params = append(preprocessing_params, list(
          no_encoding = ParamSet$new(list(ParamFct$new("encoding.branch.selection", "stability.nop"))),
          one_hot_encoding = ParamSet$new(list(ParamFct$new("encoding.branch.selection", "stability.encode")))
          ))
      }

      row_names = character()
      for (index in seq_along(preprocessing_params)) {
        model = AutoTuner$new(
          GraphLearner$new(base_pipeline, id = "feature_preprocessing"),
          resampling = rsmp("holdout"),
          measure = self$measure,
          search_space = preprocessing_params[[index]],
          terminator = trm("evals", n_evals = 1),
          tuner = tnr("random_search")
        )
        model$encapsulate = c(train = "callr", predict = "callr")
        model$train(self$task)
        if (length(model$errors) == 0) {
          output_task = get(last_pipeop, model$learner$model)$train_task
          numeric_cols = nrow(output_task$feature_types[output_task$feature_types$type %in% c("numeric", "integer"), ])
          all_cols = output_task$ncol - 1
          result = rbind(result, c(numeric_cols, all_cols))
          row_names = c(row_names, names(preprocessing_params)[[index]])
        } else {
          print(model$errors)
          print(model$learner$graph$ids())
          print(model$instance_args$search_space)
        }
      }
      rownames(result) = row_names
      print(result)
      return(result)
    },
    .extend_preprocessing = function(current_pipeline) {
      if ("imputation.imputehist" %in% current_pipeline$ids())
      replace_existing_node(current_pipeline,
                            existing_pipeop = "imputation.imputehist",
                            pipeop_choices =  c("imputation.imputehist", "imputation.imputemean", "imputation.imputemedian"),
                            branching_prefix = "numeric.",
                            columns = c("integer", "numeric"))

      if ("imputation.imputeoor" %in% current_pipeline$ids())
      replace_existing_node(current_pipeline,
                            existing_pipeop = "imputation.imputeoor",
                            pipeop_choices =  c("imputation.imputemode", "imputation.imputeoor", "imputation.imputesample"),
                            branching_prefix = "factor.",
                            columns = c("factor", "ordered", "character"))

      if ("stability.encodeimpact" %in% current_pipeline$ids())
      replace_existing_node(current_pipeline,
                            existing_pipeop = "stability.encodeimpact",
                            pipeop_choices =  c("stability.encode", "stability.encodeimpact", "stability.nop"),
                            branching_prefix = "encoding.",
                            columns = c("integer", "numeric", "factor", "ordered", "character"))
    }
  )
)
