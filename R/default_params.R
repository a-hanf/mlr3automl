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
#' @return
#' `paradox::ParamSet` containing the search space for the AutoML system
default_params = function(learner_list, task_type, num_effective_vars) {
  # model is selected during tuning as a branch of the GraphLearner
  ps = ParamSet$new()

  ps$add(
    ParamDbl$new("subsample.frac", lower = 0.1, upper = 1, tags = "budget")
  )

  # update parameter set for all known learners
  if (any(grepl("xgboost", learner_list))) {
    ps = add_xgboost_params(ps, task_type)
  }

  if (any(grepl("cv_glmnet", learner_list))) {
    ps = add_glmnet_params(ps, task_type)
  }

  if (any(grepl("svm", learner_list))) {
    ps = add_svm_params(ps, task_type)
  }

  if (any(grepl("liblinear", learner_list))) {
    ps = add_liblinear_params(ps, task_type)
  }

  if (any(grepl("ranger", learner_list))) {
    ps = add_ranger_params(ps, task_type)
  }

  # trafo function can be safely set, if parameters are not used nothing happens
  ps$trafo = function(x, param_set) {
    x = xgboost_trafo(x, param_set, task_type)
    x = ranger_trafo(x, param_set, task_type, num_effective_vars)
    x = svm_trafo(x, param_set, task_type)
    x = liblinear_trafo(x, param_set, task_type)
  }

  if (length(learner_list) > 1) {
    ps$add(ParamFct$new("branch.selection", learner_list))
    # add dependencies for branch selection
    for (learner in sub(paste0(task_type, "."), "", learner_list)) {
      for (param in ps$ids(tags = learner)) {
        ps$add_dep(param, "branch.selection",
                   CondEqual$new(paste(task_type, learner, sep = ".")))
      }
    }
  }

  return(ps)
}

# Parameter Transformation for XGBoost
xgboost_trafo = function(x, param_set, task_type) {
  transformed_params = c("xgboost.eta", "xgboost.alpha", "xgboost.lambda",
                         "xgboost.rate_drop")
  transformed_params = paste(task_type, transformed_params, sep = ".")

  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = 10^(x[[param]])
    }
  }
  return(x)
}

# XGBoost parameters
add_xgboost_params = function(param_set, task_type) {
  param_set$add(ParamSet$new(list(
    # choice of boosting algorithm
    ParamFct$new(paste(task_type, "xgboost.booster", sep = "."),
                 c("gbtree", "gblinear", "dart"), default = "gbtree", tags = "xgboost"),
    # additional parameters for dart
    ParamFct$new(paste(task_type, "xgboost.sample_type", sep = "."),
                 c("uniform", "weighted"), default = "uniform", tags = "xgboost"),
    ParamFct$new(paste(task_type, "xgboost.normalize_type", sep = "."),
                 c("tree", "forest"), default = "tree", tags = "xgboost"),
    ParamDbl$new(paste(task_type, "xgboost.rate_drop", sep = "."),
                 lower = -11, upper = 0, default = 0, tags = "xgboost"), # transformed with 10^x

    # learning rate
    ParamDbl$new(paste(task_type, "xgboost.eta", sep = "."),
                 lower = -4, upper = 0, default = -0.5, tags = "xgboost"), # transformed with 10^x

    # fidelity parameters
    ParamInt$new(paste(task_type, "xgboost.nrounds", sep = "."),
                 lower = 1, upper = 1000, default = 1, tags = "xgboost"),

    # regularization parameters
    ParamDbl$new(paste(task_type, "xgboost.alpha", sep = "."),
                 lower = -11, upper = -2, default = -11, tags = "xgboost"), # transformed with 10^x
    ParamDbl$new(paste(task_type, "xgboost.lambda", sep = "."),
                 lower = -11, upper = -2, default = -11, tags = "xgboost"), # transformed with 10^x

    # subsampling parameters
    ParamDbl$new(paste(task_type, "xgboost.subsample", sep = "."),
                 lower = 0.1, upper = 1, default = 1, tags = "xgboost"),
    ParamDbl$new(paste(task_type, "xgboost.colsample_bytree", sep = "."),
                 lower = 0.1, upper = 1, default = 1, tags = "xgboost"),
    ParamDbl$new(paste(task_type, "xgboost.colsample_bylevel", sep = "."),
                 lower = 0.1, upper = 1, default = 1, tags = "xgboost"),

    # stopping criteria
    ParamInt$new(paste(task_type, "xgboost.max_depth", sep = "."),
                 lower = 1, upper = 20, default = 6, tags = "xgboost"),
    ParamInt$new(paste(task_type, "xgboost.min_child_weight", sep = "."),
                 lower = 1, upper = 20, default = 1, tags = "xgboost")
  )))

  # additional dependencies for parameters of dart booster
  dart_params = c("xgboost.sample_type", "xgboost.rate_drop",
                  "xgboost.normalize_type")
  for (param in dart_params) {
    param_set$add_dep(paste(task_type, param, sep = "."),
                      paste(task_type, "xgboost.booster", sep = "."),
                      CondEqual$new("dart"))
  }

  # dependencies for dart, gbtree booster
  dart_gbtree_params = c("xgboost.colsample_bylevel", "xgboost.colsample_bytree",
                         "xgboost.max_depth", "xgboost.min_child_weight",
                         "xgboost.subsample")
  for (param in dart_gbtree_params) {
    param_set$add_dep(paste(task_type, param, sep = "."),
                      paste(task_type, "xgboost.booster", sep = "."),
                      CondAnyOf$new(c("dart", "gbtree")))
  }

  return(param_set)
}

# Parameter transformations for Random Forest
ranger_trafo = function(x, param_set, task_type, num_effective_vars = 1) {

  transformed_param = paste(task_type, "ranger.mtry", sep = ".")
  if (transformed_param %in% names(x)) {
    proposed_mtry = as.integer(num_effective_vars^x[[transformed_param]])
    x[[transformed_param]] = max(1, proposed_mtry)
  }

  return(x)
}

# Random Forest parameters
add_ranger_params = function(param_set, task_type) {
  param_set$add(
    ParamDbl$new(paste(task_type, "ranger.mtry", sep = "."),
                 lower = 0.1, upper = 0.9, tags = "ranger"))

  return(param_set)
}

# glmnet parameters for logistic / linear regression
add_glmnet_params = function(param_set, task_type) {
  param_set$add(
    ParamDbl$new(paste(task_type, "cv_glmnet.alpha", sep = "."),
                 lower = 0, upper = 1, default = 0, tags = "cv_glmnet"))

  return(param_set)
}

# Parameter transformations for e1071 SVM
svm_trafo = function(x, param_set, task_type) {
  transformed_params = c("svm.cost", "svm.gamma")
  transformed_params = paste(task_type, transformed_params, sep = ".")

  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = 2^(x[[param]])
    }
  }
  return(x)
}

# e1071 SVM parameters
add_svm_params = function(param_set, task_type) {
  param_set$add(ParamSet$new(list(
    # kernel is always radial, other kernels are rarely better in our experience
    ParamFct$new(paste(task_type, "svm.kernel", sep = "."),
                 c("radial"), default = "radial", tags = "svm"),
    ParamDbl$new(paste(task_type, "svm.cost", sep = "."),
                 lower = -12, upper = 12, default = 0, tags = "svm"),
    ParamDbl$new(paste(task_type, "svm.gamma", sep = "."),
                 lower = -12, upper = 12, default = 0, tags = "svm")
  )))

  if (task_type == "classif") {
    param_set$add(
      ParamFct$new(paste(task_type, "svm.type", sep = "."),
                   c("C-classification"), default = "C-classification",
                   tags = "svm"))
  } else {
    param_set$add(
      ParamFct$new(paste(task_type, "svm.type", sep = "."),
                   c("eps-regression"), default = "eps-regression",
                   tags = "svm"))
  }

  return(param_set)
}

# Parameter transformations for liblinear learners
liblinear_trafo = function(x, param_set, task_type) {
  for (param in names(x)) {
    if (grepl("liblinear.*cost", param)) {
      x[[param]] = 2^(x[[param]])
    }
    if (grepl("liblinear.*type", param)) {
      x[[param]] = as.integer(x[[param]])
    }
  }
  return(x)
}

# liblinear parameters for SVM, logistic regression and Support Vector Regression
add_liblinear_params = function(param_set, task_type) {
  param_set$add(ParamDbl$new(paste(task_type, "liblinear.cost", sep = "."),
                             lower = -10, upper = 3, default = 0, tags = "liblinear"))

  # for documentation on the types, see
  # https://www.rdocumentation.org/packages/LiblineaR/versions/2.10-8/topics/LiblineaR
  if (task_type == "classif") {
    param_set$add(ParamFct$new("classif.liblinear.type",
                               c("0", "6", "7"), default = "0", tags = "liblinear"))
  } else {
    param_set$add(ParamFct$new("regr.liblinear.type",
                               c("11", "12", "13"), default = "11", tags = "liblinear"))
  }

  return(param_set)
}
