#' @title AutoMLBase
#'
#' @description
#' Base class for AutoML in mlr3automl. Has subclasses for Classification and Regression.
#'
#' @section Internals:
#' The AutoMLBase class uses [mlr3pipelines] to create a machine learning pipeline. \cr
#' This pipeline contains multiple models (Logistic Regression, Random Forest, Gradient Boosting),
#' which are wrapped in a [GraphLearner][mlr3pipelines::GraphLearner]. \cr
#' This [GraphLearner][mlr3pipelines::GraphLearner] is wrapped in an [AutoTuner][mlr3tuning::AutoTuner] for Hyperparameter Optimization and proper resampling. \cr
#' Tuning is performed using [Hyperband][mlr3hyperband].
#'
#' @section Construction:
#' Objects should be created using the [AutoML][mlr3automl::AutoML] interface function.
#' ```
#' model = AutoML(task, learner_list, learner_timeout, resampling, measure, runtime,
#'                terminator, preprocessing, portfolio)
#' ```
#'
#' @param task ([`Task`][mlr3::Task]) \cr
#' Contains the task to be solved. Currently [`TaskClassif`][mlr3::TaskClassif] and [`TaskRegr`][mlr3::TaskRegr] are supported.
#' @param learner_list (`list()` | `character()`) \cr
#' `List` of names from [mlr_learners][mlr3::mlr_learners]. Can be used to customize the learners to be tuned over. \cr
#' Default learners for classification: `c("classif.ranger", "classif.xgboost", "classif.liblinear")` \cr
#' Default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")` \cr
#' Might break mlr3automl if a user-provided learner is incompatible with the provided task.
#' @param learner_timeout (`integer(1)`) \cr
#' Budget (in seconds) for a single parameter evaluation during model training. \cr
#' If this budget is exceeded, the evaluation is stopped and performance measured with the fallback
#' [LearnerClassifFeatureless][mlr3::LearnerClassifFeatureless] or [LearnerRegrFeatureless][mlr3::LearnerRegrFeatureless]. \cr
#' When this is `NULL` (default), the learner timeout defaults to `runtime / 5`.
#' @param resampling ([Resampling][mlr3::Resampling]) \cr
#' Contains the resampling method to be used for hyper-parameter optimization.
#' Defaults to [ResamplingHoldout][mlr3::ResamplingHoldout].
#' @param measure ([Measure][mlr3::Measure]) \cr
#' Contains the performance measure, for which we optimize during training. \cr
#' Defaults to [Accuracy][mlr3measures::acc] for classification and [RMSE][mlr3measures::rmse] for regression.
#' @param runtime (`integer(1)`) \cr
#' Number of seconds for which to run the optimization. Does *not* include training time of the final model. \cr
#' Defaults to `Inf`, letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @param terminator ([Terminator][bbotk::Terminator]) \cr
#' Contains an optional additional termination criterion for model tuning. \cr
#' Note that the [Hyperband][mlr3hyperband] tuner might stop training before the budget is exhausted.
#' [TerminatorRunTime][bbotk::TerminatorRunTime] should not be used, use the separate `runtime` parameter instead. \cr
#' Defaults to [TerminatorNone][bbotk::TerminatorNone], letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @param preprocessing (`character(1)` | [Graph][mlr3pipelines::Graph]) \cr
#' Type of preprocessing to be used. Possible values are :
#' - "none": No preprocessing at all
#' - "stability": [`pipeline_robustify`][mlr3pipelines::pipeline_robustify] is used to guarantee stability of the learners in the pipeline
#' - "full": Adds additional preprocessing operators for [Imputation][mlr3pipelines::PipeOpImpute], [Impact Encoding][mlr3pipelines::PipeOpEncodeImpact] and [PCA][mlr3pipelines::PipeOpPCA]. \cr
#' The choice of preprocessing operators is optimised during tuning.
#'
#' Alternatively, a [Graph][mlr3pipelines::Graph] object can be used to specify a custom preprocessing pipeline.
#' @param portfolio (`logical(1)`) \cr
#' `mlr3automl` tries out a fixed portfolio of known good learners prior to tuning. \cr
#' The `portfolio` parameter disables trying these portfolio learners.
#' @param additional_params ([ParamSet][paradox::ParamSet]) \cr
#' Additional parameter space to tune over, e.g. for custom learners / preprocessing. \cr
#' @param custom_trafo (`function(x, param_set)`) \cr
#' [Trafo function](https://mlr3book.mlr-org.com/searchspace.html#searchspace-trafo)
#' to be applied in addition to existing transformations. Can be used to transform
#' additional_params. \cr
#'
#' @field task ([`Task`][mlr3::Task]) \cr
#' Contains the task to be solved.
#' @field learner_list (`list()` | `character()`) \cr
#' `List` of names from [mlr_learners][mlr3::mlr_learners]. Can be used to customize the learners to be tuned over. \cr
#' @field learner_timeout (`integer(1)`) \cr
#' Budget (in seconds) for a single parameter evaluation during model training. \cr
#' If this budget is exceeded, the evaluation is stopped and performance measured with the fallback
#' [LearnerClassifFeatureless][mlr3::LearnerClassifFeatureless] or [LearnerRegrFeatureless][mlr3::LearnerRegrFeatureless]. \cr
#' When this is `NULL` (default), the learner timeout defaults to `runtime / 5`.
#' @field resampling ([Resampling][mlr3::Resampling]) \cr
#' Contains the resampling method to be used for hyper-parameter optimization.
#' @field measure ([Measure][mlr3::Measure]) \cr
#' Contains the performance measure, for which we optimize during training. \cr
#' @field learner ([AutoTuner][mlr3tuning::AutoTuner]) \cr
#' The ML pipeline at the core of mlr3automl is an [AutoTuner][mlr3tuning::AutoTuner] containing a [GraphLearner][mlr3pipelines::GraphLearner].
#' @field runtime (`integer(1)`) \cr
#' Number of seconds for which to run the optimization. Does *not* include training time of the final model. \cr
#' Defaults to `Inf`, letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @field tuning_terminator ([Terminator][bbotk::Terminator]) \cr
#' Contains an optional additional termination criterion for model tuning. \cr
#' Note that the [Hyperband][mlr3hyperband] tuner might stop training before the budget is exhausted.
#' [TerminatorRunTime][bbotk::TerminatorRunTime] should not be used, use the separate `runtime` parameter instead. \cr
#' Defaults to [TerminatorNone][bbotk::TerminatorNone], letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @field tuner ([TunerHyperband][mlr3hyperband::TunerHyperband]) \cr
#' Tuning is performed using [TunerHyperband][mlr3hyperband::TunerHyperband] with subsampling fractions between \[0.1, 1\] and \eqn{\eta = 3}
#' @field preprocessing (`character(1)` | [Graph][mlr3pipelines::Graph]) \cr
#' Type of preprocessing to be used. Possible values are :
#' - "none": No preprocessing at all
#' - "stability": [`pipeline_robustify`][mlr3pipelines::pipeline_robustify] is used to guarantee stability of the learners in the pipeline
#' - "full": Adds additional preprocessing operators for [Imputation][mlr3pipelines::PipeOpImpute], [Impact Encoding][mlr3pipelines::PipeOpEncodeImpact] and [PCA][mlr3pipelines::PipeOpPCA]. \cr
#' The choice of preprocessing operators is optimised during tuning.
#'
#' Alternatively, a [Graph][mlr3pipelines::Graph] object can be used to specify a custom preprocessing pipeline.
#' @field portfolio (`logical(1)`) \cr
#' Whether or not to try a fixed portfolio of known good learners prior to tuning. \cr
#' @field additional_params ([ParamSet][paradox::ParamSet]) \cr
#' Additional parameter space to tune over, e.g. for custom learners / preprocessing. \cr
#' @field custom_trafo (`function(x, param_set)`) \cr
#' [Trafo function](https://mlr3book.mlr-org.com/searchspace.html#searchspace-trafo)
#' to be applied in addition to existing transformations. Can be used to transform
#' additional_params. \cr
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
#' @import data.table
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
    portfolio = NULL,
    additional_params = NULL,
    custom_trafo = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #'
    #' @return [AutoMLBase][mlr3automl::AutoMLBase]
    initialize = function(task, learner_list = NULL, learner_timeout = NULL,
                          resampling = NULL, measure = NULL, runtime = Inf, terminator = NULL,
                          preprocessing = NULL, portfolio = TRUE,
                          additional_params = NULL, custom_trafo = NULL) {
      assert_task(task)
      assert_character(learner_list, any.missing = FALSE, min.len = 1)
      for (learner in learner_list) {
        assert_subset(learner, mlr_learners$keys())
      }

      if (!is.null(resampling)) assert_resampling(resampling)
      if (!is.null(measure)) assert_measure(measure)

      self$task = task
      self$resampling = resampling %??% rsmp("holdout")
      self$preprocessing = preprocessing %??% "full"
      self$additional_params = additional_params
      self$custom_trafo = custom_trafo

      self$runtime = assert_number(runtime, lower = 0)
      self$learner_timeout = assert_number(learner_timeout, lower = 0, null.ok = TRUE) %??% runtime / 5  # maybe choose a larger divisor here
      self$tuning_terminator = terminator %??% trm("none")
      self$portfolio = assert_logical(portfolio, len = 1)
      if (is.finite(self$runtime)) {
        # continue running until Hyperband training budget is used
        self$tuner = tnr("hyperband", eta = 3L, repetitions=Inf)
      } else {
        # Let Hyperband terminate training
        self$tuner = tnr("hyperband", eta = 3L)
      }
      self$learner = private$.get_default_learner()
    },
    #' @description
    #' Trains the AutoML system.
    #' @param row_ids (`integer()`)\cr
    #' Vector of training indices.
    train = function(row_ids = NULL) {
      self$learner$train(self$task, row_ids)
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
        print(self$learner$learner$errors)
      }
    },
    #' @description
    #' Returns a [Prediction][mlr3::Prediction] object for the given data based on the trained model.
    #' @param data ([data.frame] | [data.table] | [Task][mlr3::Task]) \cr
    #' New observations to be predicted. If `NULL`, defaults to the task the model
    #' was trained on.
    #' @param row_ids (`integer()`) \cr
    #' Vector of training indices.
    #' @return [`PredictionClassif`][mlr3::PredictionClassif] | [`PredictionRegr`][mlr3::PredictionRegr]
    predict = function(data = NULL, row_ids = NULL) {
      if (is.null(data)) {
        return(self$learner$predict(self$task, row_ids))
      } else {
        return(self$learner$predict(data, row_ids))
      }
    },
    #' @description
    #' Performs nested resampling. [`ResamplingHoldout`][mlr3::ResamplingHoldout] is used for the outer resampling.
    #' @return [`ResampleResult`][mlr3::ResampleResult]
    resample = function() {
      outer_resampling = rsmp("holdout")
      resample_result = mlr3::resample(self$task, self$learner, outer_resampling)
      self$learner = resample_result$learners[[1]]
      if (length(self$learner$learner$errors) > 0) {
        warning("An error occured during training. Fallback learner was used!")
        print(self$learner$learner$errors)
      }
      return(resample_result)
    },
    #' @description
    #' Helper to extract the best hyperparameters from a tuned model.
    #' @return [`data.table`][data.table::data.table]
    tuned_params = function() {
      if (is.null(self$learner$tuning_instance$archive)) {
        warning("Model has not been trained. Run the $train() method first.")
      } else {
        return(self$learner$tuning_instance$archive$best())
      }
    },
    #' @description
    #' Create explanation objects for a trained model
    #' @param iml_package (`character(0)`) \cr
    #' Package to be used: either `DALEX` or `iml`. Defaults to `DALEX`.
    #' @return explainer object
    explain = function(iml_package = "DALEX") {
      if (is.null(self$learner$tuning_instance$archive)) {
        warning("Model has not been trained. Run the $train() method first.")
      } else if(iml_package == "DALEX") {
        return(DALEXtra::explain_mlr3(
          self$learner$model$learner,
          data = self$task$data(cols = self$task$feature_names),
          y = self$task$data(cols = self$task$target_names),
          label = "mlr3automlExplainer",
          verbose = FALSE))
      } else if(iml_package == "iml") {
        return(iml::Predictor$new(
          self$learner$model$learner,
          data = self$task$data(cols = self$task$feature_names),
          y = self$task$data(cols = self$task$target_names)))
      }

      warning("requested IML package not supported. Valid choices are DALEX and iml")
    }
  ),
  private = list(
    .get_default_learner = function() {
      # number of variables is needed for setting mtry in ranger
      if (any(grepl("ranger", self$learner_list)) || (is.character(self$preprocessing) && self$preprocessing == "full")) {
        feature_counts = private$.compute_num_effective_vars()
      } else {
        feature_counts = NULL
      }

      preprocessing_pipeops = private$.get_preprocessing_pipeline()

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

      # fallback learner is featureless learner for classification / regression
      graph_learner$fallback = lrn(paste(self$task$task_type, '.featureless',
                                         sep = ""))
      # use callr encapsulation so we are able to kill model training, if it
      # takes too long
      graph_learner$encapsulate = c(train = "callr", predict = "callr")
      graph_learner$timeout = c(train = self$learner_timeout,
                                predict = self$learner_timeout)

      param_set = default_params(learner_list = self$learner_list,
                                 feature_counts = feature_counts,
                                 preprocessing = self$preprocessing,
                                 feature_types = unique(self$task$feature_types$type),
                                 additional_params = self$additional_params,
                                 custom_trafo = self$custom_trafo)

      tuner_list = list(self$tuner)

      if (self$portfolio) {
        initial_design = get_portfolio_design(self$task$task_type, param_set, self$learner_list)
        if (nrow(initial_design) > 0) {
          tuner_list = append(list(tnr("design_points", design = initial_design)), tuner_list)
        }
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
    .get_preprocessing_pipeline = function() {
      if (any(grepl("Graph|PipeOp", class(self$preprocessing)))) {
        return(self$preprocessing)
      } else  if (self$preprocessing == "none") {
        return(NULL)
      }

      # po("nop") is needed so we have a predecessor for the imputation nodes
      stability_preprocessing = po("nop", id = "start") %>>% pipeline_robustify(self$task, impute_missings = TRUE, factors_to_numeric = FALSE)
      if (any(c("factor", "ordered", "character") %in% self$task$feature_types$type)) {
        stability_preprocessing = stability_preprocessing %>>% po("encodeimpact")
      }
      stability_preprocessing$update_ids(prefix = "stability.")

      if (self$preprocessing == "stability") {
        return(stability_preprocessing)
      }

      # extended preprocessing adds more options for factor encoding
      private$.extend_preprocessing(stability_preprocessing)
      return(stability_preprocessing)
    },
    .create_robust_learner = function(learner_name) {
      # liblinear only works with columns of type double. Convert ints / bools -> dbl
      if (!all(c("integer", "logical") %in% lrn(learner_name)$feature_types)) {
        pipeline = PipeOpColApply$new(
          id = paste(learner_name, "colapply", sep="."),
          param_vals = list(
            affect_columns = selector_type(c("logical", "integer")),
            applicator = as.numeric))
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
      # create pipeline with all the PipeOps that change the number of features
      base_pipeline = pipeline_robustify(self$task, impute_missings = TRUE, factors_to_numeric = FALSE)
      output_task = base_pipeline$train(self$task)$removeconstants_postrobustify.output

      # number of features per type in task before encoding
      numeric_cols = nrow(output_task$feature_types[output_task$feature_types$type %in% c("numeric", "integer"), ])
      logical_cols = nrow(output_task$feature_types[output_task$feature_types$type %in% c("logical"), ])
      factor_cols = nrow(output_task$feature_types[output_task$feature_types$type %in% c("character", "factor", "ordered"), ])

      # if no encoding is chosen, number of columns stays the same
      result = matrix(nrow = 0, ncol = 2, byrow = TRUE)
      colnames(result) = c("numeric_cols", "all_cols")
      result = rbind(result, no_encoding = c(numeric_cols, numeric_cols + logical_cols + factor_cols))

      # 1-hot encoding creates a new column for every factor level
      # should be at most 1000 per column due to po("collapsefactors") in the pipeline
      factor_cols_one_hot = sum(sapply(output_task$levels(cols = output_task$feature_names), function(x) length(x)))
      result = rbind(result, one_hot_encoding = c(numeric_cols + factor_cols_one_hot, numeric_cols + logical_cols + factor_cols_one_hot))

      # factor encoding creates as many columns as there are target levels for every
      # categorical column for classification
      if (self$task$task_type == "classif") {
        factor_cols_impact = length(get(self$task$target_names, self$task$levels(cols = self$task$target_names))) * factor_cols
      } else {
        # for regression one column is created for every categorical feature
        factor_cols_impact = factor_cols
      }
      result = rbind(result, impact_encoding = c(numeric_cols + factor_cols_impact, numeric_cols + logical_cols + factor_cols_impact))
      print(result)
      return(result)
    },
    .extend_preprocessing = function(current_pipeline) {
      if ("stability.imputehist" %in% current_pipeline$ids())
        replace_existing_node(current_pipeline,
                              existing_pipeop = "stability.imputehist",
                              pipeop_choices =  c("stability.imputemean"),
                              branching_prefix = "numeric.",
                              columns = c("integer", "numeric"))
      if ("stability.encodeimpact" %in% current_pipeline$ids())
      replace_existing_node(current_pipeline,
                            existing_pipeop = "stability.encodeimpact",
                            pipeop_choices =  c("stability.encode", "stability.encodeimpact"),
                            branching_prefix = "encoding.",
                            columns = c("integer", "numeric", "factor", "ordered", "character"))
    }
  )
)
