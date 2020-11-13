#' @title Default parameter space for mlr3automl
#' @description
#' The parameter ranges are based on
#  https://docs.google.com/spreadsheets/d/1A8r5RgMxtRrL3nHVtFhO94DMTJ6qwkoOiakm7qj1e4g
#' @param learner_list
#' * `learner_list` :: `List` of names for `mlr3 Learners` \cr
#'   Can be used to customize the learners to be tuned over. If no parameter space
#'   is defined for the selected learner, it will be run with default parameters.
#'   Default learners for classification: `c("classif.ranger", "classif.xgboost", "classif.liblinear")`,
#'   default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")`.
#'   Might break mlr3automl if the learner is incompatible with the provided task.
#' @param task_type [`classif` | `regr`]
#' String denoting the type of task
#' @param num_effective_vars
#' Number of features after preprocessing. Used to compute `mtry` for Random Forest.
#' @param using_hyperband
#' For Tuning with Hyperband, a subsampling budget parameter is added to the pipeline.
#' @param using_prefixes
#' If `TRUE`, parameter IDs are prefixed with the `learner$id`. Used to avoid
#' name conflicts in branched pipelines.
#' @return
#' `paradox::ParamSet` containing the search space for the AutoML system
default_params = function(learner_list, feature_counts,
                          using_hyperband = TRUE, using_prefixes = TRUE,
                          preprocessing = "stability",
                          feature_types = NULL) {
  # model is selected during tuning as a branch of the GraphLearner
  param_set = ParamSet$new()
  task_type = sub("\\..*", "", learner_list[[1]])
  if (!(task_type %in% c("classif", "regr"))) {
    warning("Parameter sets have only been tested for classification and
            regression. Check your results carefully.")
  }

  param_set = add_preprocessing_params(param_set, preprocessing, using_hyperband, min(feature_counts[, "numeric_cols"]), feature_types)

  # update parameter set for all known learners
  if (any(grepl("xgboost", learner_list))) {
    param_set = add_xgboost_params(param_set, task_type, using_prefixes)
  }

  if (any(grepl("cv_glmnet", learner_list))) {
    param_set = add_glmnet_params(param_set, task_type, using_prefixes)
  }

  if (any(grepl("svm", learner_list))) {
    param_set = add_svm_params(param_set, task_type, using_prefixes)
  }

  if (any(grepl("liblinear", learner_list))) {
    param_set = add_liblinear_params(param_set, task_type, using_prefixes)
  }

  if (any(grepl("ranger", learner_list))) {
    param_set = add_ranger_params(param_set, task_type, using_prefixes)
  }

  param_set$trafo = function(x, param_set) {
    if (preprocessing == "full") {
      x = preprocessing_trafo(x, param_set, task_type, feature_counts[, "numeric_cols"])
    }

    if (any(grepl("xgboost", learner_list))) {
      x = xgboost_trafo(x, param_set, task_type, using_prefixes)
    }

    if (any(grepl("ranger", learner_list))) {
      counts = feature_counts[, "all_cols"]
      x = ranger_trafo(x, param_set, task_type, counts, using_prefixes)
    }

    if (any(grepl("svm", learner_list))) {
      x = svm_trafo(x, param_set, task_type, using_prefixes)
    }
    if (any(grepl("liblinear", learner_list))) {
      x = liblinear_trafo(x, param_set, task_type, using_prefixes)
    }
  }

  if (length(learner_list) > 1) {
    param_set$add(ParamFct$new("branch.selection", learner_list))
    # add dependencies for branch selection
    for (learner in sub(paste0(task_type, "."), "", learner_list)) {
      for (param in param_set$ids(tags = learner)) {
        param_set$add_dep(param, "branch.selection",
                   CondEqual$new(paste(task_type, learner, sep = ".")))
      }
    }
  }

  return(param_set)
}

preprocessing_trafo = function(x, param_set, task_type, num_effective_vars) {
  transformed_params = c("dimensionality.pca.rank.")
  if (!is.null(x$encoding.branch.selection) && x$encoding.branch.selection == "stability.encodeimpact") {
    effective_vars = num_effective_vars['impact_encoding']
  } else if (!is.null(x$encoding.branch.selection) && x$encoding.branch.selection == "stability.encode") {
    effective_vars = num_effective_vars['one_hot_encoding']
  } else {
    effective_vars = num_effective_vars
  }

  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = min(x[[param]], effective_vars)
    }
  }
  return(x)
}

add_preprocessing_params = function(param_set,
                                    preprocessing = "stability",
                                    using_hyperband = TRUE,
                                    count_numeric_cols = NULL,
                                    feature_types) {
  # Hyperband uses subsampling rate as a fidelity parameter
  if (using_hyperband) {
    param_set$add(
      ParamDbl$new("subsample.frac", lower = 0.1, upper = 1, tags = "budget")
    )
  }

  # add feature preprocessing
  if (preprocessing == "full") {
    # numerical imputation only happens if ints/numerical columns are present in the dataset
    if (length(intersect(c("integer", "numeric"), feature_types)) > 0) {
      param_set$add(
        # histogram imputation is removed until this issue is fixed:
        # https://github.com/mlr-org/mlr3pipelines/issues/545
        # ParamFct$new("numeric.branch.selection", c("imputation.imputehist", "imputation.imputemean", "imputation.imputemedian"), default = "imputation.imputemean"))
        ParamFct$new("numeric.branch.selection", c("imputation.imputemean", "imputation.imputemedian"), default = "imputation.imputemean"))
    }

    # factor imputation only happens if factors are present in the dataset
    if (length(intersect(c("factor", "character", "ordered"), feature_types)) > 0) {
      param_set$add(
        ParamFct$new("factor.branch.selection", c("imputation.imputeoor", "imputation.imputemode", "imputation.imputesample"), default = "imputation.imputeoor"))
    }

    # encoding always happens in robustify_pipeline
    param_set$add(
      ParamFct$new("encoding.branch.selection", c("stability.encode", "stability.encodeimpact"), default = "stability.encode"))

    # dimensionality reduction only makes sense for high dimensional data
    if (count_numeric_cols >= 2) {
      param_set$add(ParamSet$new(list(
        # ICA has been removed for now due to performance issues
        ParamFct$new("dimensionality.branch.selection", c("dimensionality.nop", "dimensionality.pca"), default = "dimensionality.nop"),
        ParamInt$new("dimensionality.pca.rank.", lower = 1, upper = count_numeric_cols, default = count_numeric_cols))))
      param_set$add_dep("dimensionality.pca.rank.", "dimensionality.branch.selection", CondEqual$new("dimensionality.pca"))
    }
  }

  return(param_set)
}


# Parameter Transformation for XGBoost
xgboost_trafo = function(x, param_set, task_type, using_prefixes) {
  transformed_params = get_transformed_param_names(
    task_type = task_type,
    learner_name = "xgboost",
    params_to_transform = c("eta", "alpha", "lambda", "rate_drop", "gamma"),
    using_prefixes = using_prefixes)

  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = 10^(x[[param]])
    }
  }
  return(x)
}

# XGBoost parameters
add_xgboost_params = function(param_set, task_type, using_prefixes) {
  param_id_prefix = get_param_id_prefix(task_type, "xgboost", using_prefixes)

  param_set$add(ParamSet$new(list(
    # choice of boosting algorithm
    ParamFct$new(paste0(param_id_prefix, "booster"),
                 c("gbtree", "gblinear", "dart"), default = "gbtree", tags = "xgboost"),
    # additional parameters for dart
    ParamFct$new(paste0(param_id_prefix, "sample_type"),
                 c("uniform", "weighted"), default = "uniform", tags = "xgboost"),
    ParamFct$new(paste0(param_id_prefix, "normalize_type"),
                 c("tree", "forest"), default = "tree", tags = "xgboost"),
    ParamDbl$new(paste0(param_id_prefix, "rate_drop"),
                 lower = -11, upper = 0, default = 0, tags = "xgboost"), # transformed with 10^x

    # learning rate
    ParamDbl$new(paste0(param_id_prefix, "eta"),
                 lower = -4, upper = 0, default = -0.5, tags = "xgboost"), # transformed with 10^x

    # fidelity parameters
    ParamInt$new(paste0(param_id_prefix, "nrounds"),
                 lower = 1, upper = 1000, default = 1, tags = "xgboost"),

    # regularization parameters
    ParamDbl$new(paste0(param_id_prefix, "alpha"),
                 lower = -11, upper = -2, default = -11, tags = "xgboost"), # transformed with 10^x
    ParamDbl$new(paste0(param_id_prefix, "lambda"),
                 lower = -11, upper = -2, default = -11, tags = "xgboost"), # transformed with 10^x

    # subsampling parameters
    ParamDbl$new(paste0(param_id_prefix, "subsample"),
                 lower = 0.5, upper = 1, default = 1, tags = "xgboost"),
    ParamDbl$new(paste0(param_id_prefix, "colsample_bytree"),
                 lower = 0.5, upper = 1, default = 1, tags = "xgboost"),
    ParamDbl$new(paste0(param_id_prefix, "colsample_bylevel"),
                 lower = 0.5, upper = 1, default = 1, tags = "xgboost"),

    # stopping criteria
    ParamInt$new(paste0(param_id_prefix, "max_depth"),
                 lower = 3, upper = 20, default = 6, tags = "xgboost"),
    ParamInt$new(paste0(param_id_prefix, "min_child_weight"),
                 lower = 1, upper = 20, default = 1, tags = "xgboost"),
    ParamDbl$new(paste0(param_id_prefix, "gamma"),
                 lower = -4, upper = 2, default = 0, tags = "xgboost") # transformed with 10^x
  )))

  # additional dependencies for parameters of dart booster
  dart_params = paste0(param_id_prefix, c("sample_type", "rate_drop", "normalize_type"))

  for (param in dart_params) {
    param_set$add_dep(param, paste0(param_id_prefix, "booster"), CondEqual$new("dart"))
  }

  # dependencies for dart, gbtree booster
  dart_gbtree_params = paste0(param_id_prefix,
                              c("colsample_bylevel", "colsample_bytree", "gamma",
                                "max_depth", "min_child_weight", "subsample"))

  for (param in dart_gbtree_params) {
    param_set$add_dep(param, paste0(param_id_prefix, "booster"),
                      CondAnyOf$new(c("dart", "gbtree")))
  }

  return(param_set)
}

# Parameter transformations for Random Forest
ranger_trafo = function(x, param_set, task_type, num_effective_vars, using_prefixes) {
  transformed_param = get_transformed_param_names(
    task_type = task_type,
    learner_name = "ranger",
    params_to_transform = "mtry",
    using_prefixes = using_prefixes)

  if (!is.null(x$dimensionality.ica.n.comp) || !is.null(x$dimensionality.pca.rank.)) {
    effective_vars = x$dimensionality.ica.n.comp %??% x$dimensionality.pca.rank.
  } else if (!is.null(x$encoding.branch.selection) && x$encoding.branch.selection == "stability.encodeimpact") {
    effective_vars = num_effective_vars['impact_encoding']
  } else if (!is.null(x$encoding.branch.selection) && x$encoding.branch.selection == "stability.encode") {
    effective_vars = num_effective_vars['one_hot_encoding']
  } else {
    effective_vars = num_effective_vars
  }

  if (transformed_param %in% names(x)) {
    proposed_mtry = as.integer(effective_vars^x[[transformed_param]])
    x[[transformed_param]] = max(1, proposed_mtry)
  }

  return(x)
}

# Random Forest parameters
add_ranger_params = function(param_set, task_type, using_prefixes) {
  param_id_prefix = get_param_id_prefix(task_type, "ranger", using_prefixes)

  param_set$add(ParamSet$new(list(
    ParamDbl$new(paste0(param_id_prefix, "mtry"),
                 lower = 0.1, upper = 0.9, default = 0.5, tags = "ranger"),
    ParamFct$new(paste0(param_id_prefix, "splitrule"),
                 c("gini", "extratrees"), default = "gini", tags = "ranger"))))
  return(param_set)
}

# glmnet parameters for logistic / linear regression
add_glmnet_params = function(param_set, task_type, using_prefixes) {
  param_id_prefix = get_param_id_prefix(task_type, "cv_glmnet", using_prefixes)

  param_set$add(ParamDbl$new(
    paste0(param_id_prefix, "alpha"),
    lower = 0, upper = 1, default = 0, tags = "cv_glmnet"))

  return(param_set)
}

# Parameter transformations for e1071 SVM
svm_trafo = function(x, param_set, task_type, using_prefixes) {
  transformed_params = get_transformed_param_names(
    task_type = task_type,
    learner_name = "svm",
    params_to_transform = c("cost", "gamma"),
    using_prefixes = using_prefixes)

  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = 2^(x[[param]])
    }
  }
  return(x)
}

# e1071 SVM parameters
add_svm_params = function(param_set, task_type, using_prefixes) {
  param_id_prefix = get_param_id_prefix(task_type, "svm", using_prefixes)

  param_set$add(ParamSet$new(list(
    # kernel is always radial, other kernels are rarely better in our experience
    ParamFct$new(paste0(param_id_prefix, "kernel"),
                 c("radial"), default = "radial", tags = "svm"),
    ParamDbl$new(paste0(param_id_prefix, "cost"),
                 lower = -12, upper = 12, default = 0, tags = "svm"),
    ParamDbl$new(paste0(param_id_prefix, "gamma"),
                 lower = -12, upper = 12, default = 0, tags = "svm")
  )))

  if (task_type == "classif") {
    param_set$add(ParamFct$new(
      paste0(param_id_prefix, "type"),
      c("C-classification"), default = "C-classification", tags = "svm"))
  } else {
    param_set$add(ParamFct$new(
      paste0(param_id_prefix, "type"),
      c("eps-regression"), default = "eps-regression", tags = "svm"))
  }

  return(param_set)
}

# Parameter transformations for liblinear learners
liblinear_trafo = function(x, param_set, task_type, using_prefixes) {
  transformed_params = get_transformed_param_names(
    task_type = task_type,
    learner_name = "liblinear",
    params_to_transform = c("cost", "type"),
    using_prefixes = using_prefixes)

  for (param in names(x)) {
    if (param %in% transformed_params && grepl("cost", param)) {
      x[[param]] = 2^(x[[param]])
    }
    if (param %in% transformed_params && grepl("type", param)) {
      x[[param]] = as.integer(x[[param]])
    }
  }
  return(x)
}

# liblinear parameters for SVM, logistic regression and Support Vector Regression
add_liblinear_params = function(param_set, task_type, using_prefixes) {
  param_id_prefix = get_param_id_prefix(task_type, "liblinear", using_prefixes)

  param_set$add(ParamDbl$new(paste0(param_id_prefix, "cost"),
                             lower = -10, upper = 3, default = 0, tags = "liblinear"))

  # for documentation on the types, see
  # https://www.rdocumentation.org/packages/LiblineaR/versions/2.10-8/topics/LiblineaR
  if (task_type == "classif") {
    param_set$add(ParamFct$new(paste0(param_id_prefix, "type"),
                               c("0", "6", "7"), default = "0", tags = "liblinear"))
  } else {
    param_set$add(ParamFct$new(paste0(param_id_prefix, "type"),
                               c("11", "12", "13"), default = "11", tags = "liblinear"))
  }

  return(param_set)
}

get_transformed_param_names = function(task_type, learner_name, params_to_transform, using_prefixes) {
  if (!using_prefixes) {
    return(params_to_transform)
  }
  return(paste(task_type, learner_name, params_to_transform, sep = "."))
}

get_param_id_prefix = function(task_type, learner_name, using_prefixes) {
  if (using_prefixes) {
    return(paste0(task_type, ".", learner_name, "."))
  }
  return("")
}
