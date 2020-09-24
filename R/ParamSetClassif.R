default_classification_params = function(learner_list) {

  # model is selected during tuning as a branch of the GraphLearner
  ps = ParamSet$new(list(ParamFct$new("branch.selection", learner_list)))

  if ("classif.xgboost" %in% learner_list) {
    ps = add_classif_xgboost_params(ps)
  }

  ps$trafo = function(x, param_set) {
    x = classif_xgboost_trafo(x, param_set)
  }

  return(ps)
}

classif_xgboost_trafo = function(x, param_set) {
  transformed_params = c("classif.xgboost.eta", "classif.xgboost.alpha",
                         "classif.xgboost.lambda", "classif.xgboost.rate_drop")
  for (param in names(x)) {
    if (param %in% transformed_params) {
      x[[param]] = 10^(x[[param]])
    } else if (param == "classif.xgboost.nrounds") {
      x[[param]] = as.integer(2^(x[[param]]))
    }
  }
  return(x)
}

add_classif_xgboost_params = function(param_set) {
  param_set$add(ParamSet$new(list(
    # choice of boosting algorithm
    ParamFct$new("classif.xgboost.booster", c("gbtree", "gblinear", "dart"),
                 default = "gbtree", tags = "xgboost"),
    # additional parameters for dart
    ParamFct$new("classif.xgboost.sample_type", c("uniform", "weighted"),
                 default = "uniform", tags = "xgboost"),
    ParamFct$new("classif.xgboost.normalize_type", c("tree", "forest"),
                 default = "tree", tags = "xgboost"),
    ParamDbl$new("classif.xgboost.rate_drop", lower = -11, upper = 0,
                 default = 0, tags = "xgboost"), # transformed with 10^x

    # learning rate
    ParamDbl$new("classif.xgboost.eta", lower = -4, upper = 0,
                 default = -0.5, tags = "xgboost"), # transformed with 10^x

    # number of boosting iterations, TODO: ask about tuning this
    ParamDbl$new("classif.xgboost.nrounds", lower = 0, upper = 9,
                 default = 0, tags = "xgboost"), # transformed with 2^x

    # regularization parameters
    ParamDbl$new("classif.xgboost.alpha", lower = -11, upper = -2,
                 default = -11, tags = "xgboost"), # transformed with 10^x
    ParamDbl$new("classif.xgboost.lambda", lower = -11, upper = -2,
                 default = -11, tags = "xgboost"), # transformed with 10^x

    # subsampling parameters
    ParamDbl$new("classif.xgboost.subsample", lower = 0.1, upper = 1,
                 default = 1, tags = "xgboost"),
    ParamDbl$new("classif.xgboost.colsample_bytree", lower = 0.1, upper = 1,
                 default = 1, tags = "xgboost"),
    ParamDbl$new("classif.xgboost.colsample_bylevel", lower = 0.1, upper = 1,
                 default = 1, tags = "xgboost"),

    # stopping criteria
    ParamInt$new("classif.xgboost.max_depth", lower = 1, upper = 20,
                 default = 6, tags = "xgboost"),
    ParamInt$new("classif.xgboost.min_child_weight", lower = 1, upper = 20,
                 default = 1, tags = "xgboost")
  )))

  # only tune over these hyperparameters if XGBoost branch is chosen
  for (param in param_set$ids(tags = "xgboost")) {
    param_set$add_dep(param, "branch.selection", CondEqual$new("classif.xgboost"))
  }

  # additional dependencies for parameters of dart booster
  dart_params = c("classif.xgboost.sample_type", "classif.xgboost.rate_drop",
                  "classif.xgboost.normalize_type")
  for (param in dart_params) {
    param_set$add_dep(param, "classif.xgboost.booster", CondEqual$new("dart"))
  }

  return(param_set)
}

classif_ranger_trafo = function(x, param_set, num_effective_vars) {
  proposed_mtry = as.integer(num_effective_vars^x$classif.ranger.mtry)
  max_features = 200
  x$classif.ranger.mtry = min(max(1, proposed_mtry), max_features)
  return(x)
  # proposed_mtry = as.integer(num_effective_vars^x$regr.ranger.mtry)
  # x$regr.ranger.mtry = min(max(1, proposed_mtry), 200)
  # return(x)
}

add_mtry_to_classif_params = function(param_set, num_effective_vars) {
  param_set$add(ParamDbl$new("classif.ranger.mtry", lower = 0.1, upper = 0.9,
                tags = "classif.ranger"))

  old_trafo_function = param_set$trafo

  param_set$trafo = function(x, param_set) {
    x = old_trafo_function(x, param_set)
    x = classif_ranger_trafo(x, param_set, num_effective_vars)
    return(x)
    # proposed_mtry = as.integer(num_effective_vars^x$regr.ranger.mtry)
    # x$regr.ranger.mtry = min(max(1, proposed_mtry), 200)
    # return(x)
  }

  param_set$add_dep("classif.ranger.mtry", "branch.selection",
                    CondEqual$new("classif.ranger"))

  return(param_set)
}

