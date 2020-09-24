default_regression_params = function(learner_list) {

  # model is selected during tuning as a branch of the GraphLearner
  ps = ParamSet$new(list(ParamFct$new("branch.selection", learner_list)))

  if ("regr.xgboost" %in% learner_list) {
    ps = add_regr_xgboost_params(ps)
  }

  ps$trafo = function(x, param_set) {
    x = regr_xgboost_trafo(x, param_set)
  }

  return(ps)
}

regr_xgboost_trafo = function(x, param_set) {
  transformed_params = c("regr.xgboost.eta", "regr.xgboost.alpha",
                         "regr.xgboost.lambda", "regr.xgboost.rate_drop")
  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = 10^(x[[param]])
    } else if (param == "regr.xgboost.nrounds") {
      x[[param]] = as.integer(2^(x[[param]]))
    }
  }
  return(x)
}

add_regr_xgboost_params = function(param_set) {
  param_set$add(ParamSet$new(list(
    # choice of boosting algorithm
    ParamFct$new("regr.xgboost.booster", c("gbtree", "gblinear", "dart"),
                 default = "gbtree", tags = "xgboost"),
    # additional parameters for dart
    ParamFct$new("regr.xgboost.sample_type", c("uniform", "weighted"),
                 default = "uniform", tags = "xgboost"),
    ParamFct$new("regr.xgboost.normalize_type", c("tree", "forest"),
                 default = "tree", tags = "xgboost"),
    ParamDbl$new("regr.xgboost.rate_drop", lower = -11, upper = 0,
                 default = 0, tags = "xgboost"), # transformed with 10^x

    # learning rate
    ParamDbl$new("regr.xgboost.eta", lower = -4, upper = 0,
                 default = -0.5, tags = "xgboost"), # transformed with 10^x

    # number of boosting iterations, TODO: ask about tuning this
    ParamDbl$new("regr.xgboost.nrounds", lower = 0, upper = 9,
                 default = 0, tags = "xgboost"), # transformed with 2^x

    # regularization parameters
    ParamDbl$new("regr.xgboost.alpha", lower = -11, upper = -2,
                 default = -11, tags = "xgboost"), # transformed with 10^x
    ParamDbl$new("regr.xgboost.lambda", lower = -11, upper = -2,
                 default = -11, tags = "xgboost"), # transformed with 10^x

    # subsampling parameters
    ParamDbl$new("regr.xgboost.subsample", lower = 0.1, upper = 1,
                 default = 1, tags = "xgboost"),
    ParamDbl$new("regr.xgboost.colsample_bytree", lower = 0.1, upper = 1,
                 default = 1, tags = "xgboost"),
    ParamDbl$new("regr.xgboost.colsample_bylevel", lower = 0.1, upper = 1,
                 default = 1, tags = "xgboost"),

    # stopping criteria
    ParamInt$new("regr.xgboost.max_depth", lower = 1, upper = 20,
                 default = 6, tags = "xgboost"),
    ParamInt$new("regr.xgboost.min_child_weight", lower = 1, upper = 20,
                 default = 1, tags = "xgboost")
  )))

  # only tune over these hyperparameters if XGBoost branch is chosen
  for (param in param_set$ids(tags = "xgboost")) {
    param_set$add_dep(param, "branch.selection", CondEqual$new("regr.xgboost"))
  }

  # additional dependencies for parameters of dart booster
  dart_params = c("regr.xgboost.sample_type", "regr.xgboost.rate_drop",
                  "regr.xgboost.normalize_type")
  for (param in dart_params) {
    param_set$add_dep(param, "regr.xgboost.booster", CondEqual$new("dart"))
  }

  return(param_set)
}

regr_ranger_trafo = function(x, param_set, num_effective_vars) {
  proposed_mtry = as.integer(num_effective_vars^x$regr.ranger.mtry)
  max_features = 200
  x$regr.ranger.mtry = min(max(1, proposed_mtry), max_features)
  return(x)
  # proposed_mtry = as.integer(num_effective_vars^x$regr.ranger.mtry)
  # x$regr.ranger.mtry = min(max(1, proposed_mtry), 200)
  # return(x)
}

add_mtry_to_regr_params = function(param_set, num_effective_vars) {
  param_set$add(ParamDbl$new("regr.ranger.mtry", lower = 0.1, upper = 0.9,
                             tags = "regr.ranger"))

  old_trafo_function = param_set$trafo

  param_set$trafo = function(x, param_set) {
    x = old_trafo_function(x, param_set)
    x = regr_ranger_trafo(x, param_set, num_effective_vars)
    return(x)
    # proposed_mtry = as.integer(num_effective_vars^x$regr.ranger.mtry)
    # x$regr.ranger.mtry = min(max(1, proposed_mtry), 200)
    # return(x)
  }

  param_set$add_dep("regr.ranger.mtry", "branch.selection",
                    CondEqual$new("regr.ranger"))

  return(param_set)
}
