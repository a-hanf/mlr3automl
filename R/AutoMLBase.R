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
#' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
#'   Contains the resampling method to be used for hyper-parameter optimization
#' * `measure` :: `Measure` object from `mlr_measures` \cr
#'   Contains the performance measure, for which we optimize during training
#' * `tuning_terminator` :: `Terminator` object from `mlr3tuning` \cr
#'   Contains the termination criterion for model tuning
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
    resampling = NULL,
    measure = NULL,
    tuning_terminator = NULL,
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
    #'   Budget (in seconds) for a single learner during training of the pipeline.
    #'   If this budget is exceeded, the learner is replaced with the fallback
    #'   learner (`lrn("classif.featureless")` or `lrn("regr.featureless")`).
    #' @param resampling
    #' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
    #'   Contains the resampling method to be used for hyper-parameter optimization.
    #'   Defaults to `rsmp("holdout")`.
    #' @param measure
    #' * `measure` :: `Measure` object from `mlr_measures` \cr
    #'   Contains the performance measure, for which we optimize during training.
    #'   Defaults to `msr("classif.acc")` for classification and `msr("regr.rmse")`
    #'   for regression.
    #' @param terminator
    #' * `terminator` :: `Terminator` object from `mlr3tuning` \cr
    #'   Contains the termination criterion for model tuning. Note that the Hyperband
    #'   tuner might stop training before the budget is exhausted.
    #'   Defaults to `trm("none")`
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, terminator = NULL) {

      assert_task(task)
      assert_character(learner_list, any.missing = FALSE, min.len = 1)
      for (learner in learner_list) {
        assert_subset(learner, mlr_learners$keys())
      }
      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measure)) assert_measure(measure)

      self$task = task
      self$resampling = resampling %??% rsmp("holdout")

      if (is.null(learner_timeout)) {
        if (!is.null(terminator$param_set$values$secs)) {
          learner_timeout = as.integer(terminator$param_set$values$secs / 5)
        } else {
          learner_timeout = Inf
        }
      }

      self$learner_timeout = learner_timeout
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
      if (is.null(self$learner$state)) {
        warning("Model has not been trained. Run the $train() method first.")
      } else {
        return(self$learner$tuning_instance$archive$best())
      }
    }
  ),
  private = list(
    .get_default_learner = function() {
      learners = list()
      for (learner in self$learner_list) {
        learners = append(learners, private$.create_robust_learner(learner))
      }
      names(learners) = self$learner_list
      if (length(self$learner_list) > 1) {
        pipeline = po("subsample") %>>% ppl("branch", graphs = learners)
      } else {
        pipeline = po("subsample") %>>% learners[[1]]
      }

      graph_learner = GraphLearner$new(pipeline)

      if (!is.null(self$learner_timeout) || !is.infinite(self$learner_timeout)) {
        # fallback learner is featureless learner for classification / regression
        graph_learner$fallback = lrn(paste(self$task$task_type, '.featureless',
                                           sep = ""))
        # use callr encapsulation so we are able to kill model training, if it
        # takes too long
        graph_learner$encapsulate = c(train = "callr", predict = "callr")
        graph_learner$timeout = c(train = self$learner_timeout, predict = self$learner_timeout)
      }

      if (any(grepl("ranger", self$learner_list))) {
        num_effective_vars = private$.compute_num_effective_vars()
      } else {
        num_effective_vars = NULL
      }

      param_set = default_params(self$learner_list, self$task$task_type, num_effective_vars)

      return(AutoTuner$new(
        learner = graph_learner,
        resampling = self$resampling,
        measure = self$measure,
        search_space = param_set,
        terminator = self$tuning_terminator,
        tuner = self$tuner))
    },
    .create_robust_learner = function(learner_name) {
      # temporary workaround, see https://github.com/mlr-org/mlr3pipelines/issues/519
      pipeline = po("nop")

      # robustify_pipeline takes care of imputation, factor encoding etc.
      # we always need imputation, because earlier preprocessing pipeops may introduce missing values
      pipeline = pipeline %>>%
        pipeline_robustify(task = self$task, learner = lrn(learner_name),
                           impute_missings = TRUE)

      # liblinear only works with columns of type double. Convert ints / bools -> dbl
      if (grepl('liblinear', learner_name)) {
        pipeline = pipeline %>>%
          po("colapply", applicator = as.numeric,
             param_vals = list(affect_columns = selector_type(c("logical", "integer"))))
      }

      # avoid name conflicts in pipeline
      pipeline$update_ids(prefix = paste0(learner_name, "."))

      # liblinear learner offer logistic/linear regression as well as SVMs
      # SVMs do not offer probability predictions and can not be tuned for AUC
      # thus, only use logistic regression for now
      # if (grepl('liblinear', learner_name) && self$task$task_type == "classif") {
      #   liblinear_learners = list(
      #     po("learner", lrn(learner_name, predict_type = "prob"), id = paste(learner_name, "logreg", sep = ".")),
      #     po("learner", lrn(learner_name, predict_type = "response"), id = paste(learner_name, "svm", sep = ".")))
      #   choices = c("classif.liblinear.logreg", "classif.liblinear.svm")
      #   return(
      #     pipeline %>>%
      #     po("branch", choices, id = "classif.liblinear.branch") %>>%
      #     gunion(graphs = liblinear_learners) %>>%
      #     po("unbranch", choices, id = "classif.liblinear.unbranch"))
      # }

      # predict probabilities for classification if possible
      if (self$task$task_type == "classif" && ("prob" %in% lrn(learner_name)$predict_types)) {
        return(pipeline %>>% po("learner", lrn(learner_name, predict_type = "prob")))
      }
      # default: predict with type response
      return(pipeline %>>% po("learner", lrn(learner_name)))
    },
    .compute_num_effective_vars = function() {
      rf_learner = lrn(paste(self$task$task_type, 'ranger', sep = "."))

      pipeline =
        po("nop") %>>%
        pipeline_robustify(task = self$task, learner = rf_learner, impute_missings = TRUE) %>>%
        lrn(paste(self$task$task_type, '.featureless', sep = ""))
      pipeline$train(self$task)

      # get number of variables after preprocessing
      last_pipeop = paste(self$task$task_type, '.featureless', sep = "")
      num_effective_vars = get(last_pipeop, pipeline$state)$train_task$ncol - 1
      return(num_effective_vars)
    }
  )
)
