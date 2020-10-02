default_params = function(learner_list, task_type) {

  # model is selected during tuning as a branch of the GraphLearner
  ps = ParamSet$new(list(ParamFct$new("branch.selection", learner_list)))

  # update parameter set for all known learners
  if (any(grepl("xgboost", learner_list))) {
    ps = add_xgboost_params(ps, task_type)
  }

  # trafo function can be safely set, if parameters are not used nothing happens
  ps$trafo = function(x, param_set) {
    x = xgboost_trafo(x, param_set, task_type)
  }

  return(ps)
}

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
                 lower = 10^6, upper = 10^6, default = 10^6, tags = "xgboost"),
    ParamInt$new(paste(task_type, "xgboost.early_stopping_rounds", sep = "."),
                 lower = 1, upper = 50, default = 10, tags = "xgboost"),

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

  # only tune over these hyperparameters if XGBoost branch is chosen
  for (param in param_set$ids(tags = "xgboost")) {
    param_set$add_dep(param, "branch.selection",
                      CondEqual$new(paste(task_type, "xgboost", sep = ".")))
  }

  # additional dependencies for parameters of dart booster
  dart_params = c("xgboost.sample_type", "xgboost.rate_drop",
                  "xgboost.normalize_type")
  for (param in dart_params) {
    param_set$add_dep(paste(task_type, param, sep = "."),
                      paste(task_type, "xgboost.booster", sep = "."),
                      CondEqual$new("dart"))
  }

  return(param_set)
}

ranger_trafo = function(x, param_set, num_effective_vars, task_type) {
  proposed_mtry = as.integer(num_effective_vars^x$regr.ranger.mtry)
  max_features = 200
  x[[paste(task_type, "ranger.mtry", sep = ".")]] =
    min(max(1, proposed_mtry), max_features)
  return(x)
}

add_mtry_to_ranger_params = function(param_set, num_effective_vars, task_type) {
  param_set$add(ParamDbl$new(paste(task_type, "ranger.mtry", sep = "."),
                             lower = 0.1, upper = 0.9, tags = "ranger"))

  old_trafo_function = param_set$trafo

  param_set$trafo = function(x, param_set) {
    x = old_trafo_function(x, param_set)
    x = ranger_trafo(x, param_set, num_effective_vars, task_type)
    return(x)
  }

  param_set$add_dep(paste(task_type, "ranger.mtry", sep = "."),
                    "branch.selection",
                    CondEqual$new(paste(task_type, "ranger", sep = ".")))

  return(param_set)
}
