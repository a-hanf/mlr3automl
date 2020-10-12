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
#' * `learner` :: `GraphLearner` object from `mlr3pipelines` \cr
#'   Contains the machine learning pipeline with preprocessing and multiple learners
#' * `resampling` :: `Resampling` object from `mlr3tuning` \cr
#'   Contains the resampling method to be used for hyper-parameter optimization
#' * `measures` :: `Measure` object from `mlr_measures` \cr
#'   Contains the performance measure, for which we optimize during training
#' * `param_set` :: `ParamSet` object from `paradox` \cr
#'   Contains the parameter space over which we optimize
#' * `tuning_terminator` :: `Terminator` object from `mlr3tuning` \cr
#'   Contains the termination criterion for model tuning
#' * `tuner` :: `Tuner` object from `mlr3tuning` \cr
#'   Contains the tuning strategy used for hyper-parameter optimization (default: Random Search)
#'   Note: We will be switching to Hyperband for the tuning as soon as it works.
#' @section Methods
#' * `train()` \cr
#'   Runs the AutoML system. The trained model is saved in the $learner slot.
#' * `predict(data = NULL, row_ids = NULL)` \cr
#'   `data.frame | data.table | Task -> PredictionClassif or PredictionRegr`
#'   Returns a Prediction object for the given data based on the trained model.
#'   If data is NULL, defaults to the task used for training
#'   `resample(outer_resampling_holdout_ratio = 0.8)`
#'   `double(1) -> ResampleResult`
#'   Performs nested resampling with a train/test split as the outer resampling
#' @import checkmate
#' @import mlr3
#' @import mlr3learners
#' @import mlr3oml
#' @import mlr3pipelines
#' @import mlr3tuning
#' @import paradox
#' @import testthat
#' @importFrom R6 R6Class
#' @export
#' @name AutoMLBase
#' @examples
#' "add later"
AutoMLBase = R6Class("AutoMLBase",
  public = list(
    task = NULL,
    learner_list = NULL,
    learner = NULL,
    resampling = NULL,
    measures = NULL,
    param_set = NULL,
    tuning_terminator = NULL,
    tuner = NULL,
    initialize = function(task, learner_list = NULL, resampling = NULL,
                          measures = NULL, terminator = NULL) {
      assert_task(task)
      for (learner in learner_list) {
        testthat::expect_true(learner %in% mlr_learners$keys())
      }
      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measures)) assert_measures(measures)
      # FIXME: find / write assertion for terminator class
      # if (!is.null(terminator)) assert_terminator(terminator)
      self$task = task
      self$resampling = resampling %??% rsmp("cv", folds = 3)
      self$tuning_terminator = terminator %??%
        trm('combo', list(trm('run_time', secs = 60), trm('stagnation')))
      self$tuner = tnr("random_search")
      self$param_set = private$.get_default_param_set()
      self$learner = private$.get_default_learner()
    },
    train = function(row_ids = NULL) {
      self$learner$train(self$task, row_ids)
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
        print(self$learner$learner$errors)
      }
    },
    predict = function(data = NULL, row_ids = NULL) {
      if (is.null(data)) {
        return(self$learner$predict(self$task, row_ids))
      } else {
        return(self$learner$predict(data, row_ids))
      }
    },
    resample = function() {
      outer_resampling = rsmp("holdout")
      resample_result = mlr3::resample(self$task, self$learner,
                                       outer_resampling, store_models = TRUE)
      self$learner = resample_result$learners[[1]]
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
      }
      return(resample_result)
    },
    tuned_params = function() {
      return(self$learner$tuning_instance$archive$best())
    }
  ),
  private = list(
    .get_default_learner = function() {
      learners = list()
      for (learner in self$learner_list) {
        learners = append(learners, private$.create_robust_learner(learner))
      }
      names(learners) = self$learner_list
      pipeline = ppl("branch", graphs = learners)
      graph_learner = GraphLearner$new(pipeline)

      # # fallback learner is featureless learner for classification / regression
      # graph_learner$fallback = lrn(paste(self$task$task_type, '.featureless',
      #                                    sep = ""))
      # # use callr encapsulation so we are able to kill model training, if it
      # # takes too long
      # graph_learner$encapsulate = c(train = "callr", predict = "callr")

      return(AutoTuner$new(graph_learner, self$resampling, self$measures,
                           self$param_set, self$tuning_terminator, self$tuner))
    },
    .create_robust_learner = function(learner_name) {
      # Tree-based methods can handle factors and missing values natively
      if (grepl("ranger", learner_name) || grepl("xgboost", learner_name)) {
        pipeline = pipeline_robustify(task = self$task,
                                      learner = lrn(learner_name))
        # SVMs from e1071 and liblinear need imputation / encoding
        # also good default setting for learners with unconfigured param spaces
      } else {
        pipeline = pipeline_robustify(task = self$task,
                                      learner = lrn(learner_name),
                                      impute_missings = TRUE,
                                      factors_to_numeric = TRUE)
      }
      # avoid name conflicts in pipeline
      pipeline$set_names(pipeline$ids(),
                         paste(learner_name, pipeline$ids(), sep = "."))
      # mtry is set at runtime depending on the number of features
      if (grepl('ranger', learner_name)) {
        private$.set_mtry_for_random_forest(pipeline)
      }

      # liblinear only works with columns of type double. Convert ints / bools -> dbl
      if (grepl('liblinear', learner_name)) {
        pipeline = pipeline %>>%
          po("colapply", applicator = as.numeric,
             param_vals = list(affect_columns = selector_type(c("logical", "integer"))))
      }

      # predict probabilities for classification if possible
      if (self$task$task_type == "classif" && ("prob" %in% lrn(learner_name)$predict_types)) {
        return(pipeline %>>% po("learner", lrn(learner_name, predict_type = "prob")))
      }
      # default: predict with type response
      return(pipeline %>>% po("learner", lrn(learner_name)))
    },
    .set_mtry_for_random_forest = function(pipeline) {
      # deep copy so we don't mess up the original pipeline
      pipe_copy = pipeline$clone(deep = TRUE) %>>%
        lrn(paste(self$task$task_type, '.featureless', sep = ""))
      # train without an informative learner so we can see the output of the
      # preprocessing pipeline
      pipe_copy$train(self$task)
      last_pipeop = pipe_copy$ids()[length(pipe_copy$ids())]
      # get number of variables after encoding from input of final pipeop
      num_effective_vars = length(get(last_pipeop, pipe_copy$state)$train_task$feature_names)
      self$param_set = add_mtry_to_ranger_params(
        self$param_set, num_effective_vars, self$task$task_type)
    },
    .get_default_param_set = function() {
      ps = default_params(self$learner_list, self$task$task_type)
      return(ps)
    }
  )
)

#' @title Interface function for mlr3automl
#'
#' @description
#' Interface for classes AutoMLClassif and AutoMLRegr.
#' @inheritParams AutoMLBase
#' @return ['AutoMLClassif' | 'AutoMLRegr']
#' @export
#' @examples
#' \dontrun{
#' automl_object = AutoML(tsk("iris"))
#' }
AutoML = function(task, learner_list = NULL, resampling = NULL, measures = NULL,
                  terminator = NULL) {
  if (task$task_type == "classif") {
    # stratify target variable so that every target label appears
    # in all folds while resampling
    target_is_factor = task$col_info[task$col_info$id == task$target_names, ]$type == "factor"
    if (length(target_is_factor) == 1 && target_is_factor) {
      task$col_roles$stratum = task$target_names
    }
    return(AutoMLClassif$new(task, learner_list, resampling, measures,
                             terminator))
  } else if (task$task_type == "regr") {
    return(AutoMLRegr$new(task, learner_list, resampling, measures, terminator))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
