default_regression_params = function() {
  ps = ParamSet$new(list(
    ParamInt$new("regr.xgboost.max_depth", lower = 1, upper = 20, default = 6, tags = "xgboost"),
    ParamDbl$new("regr.xgboost.eta", lower = 10^-4, upper = 10^0, default = 0.3, tags = "xgboost"), # needs trafo
    ParamInt$new("regr.xgboost.nrounds", lower = 512, upper = 512, default = 512, tags = "xgboost"),
    ParamFct$new("regr.xgboost.booster", c("gbtree", "gblinear", "dart"), default = "gbtree", tags = "xgboost"),
    ParamDbl$new("regr.xgboost.subsample", lower = 0.1, upper = 1, default = 1, tags = "xgboost"),
    ParamInt$new("regr.xgboost.min_child_weight", lower = 1, upper = 20, default = 1, tags = "xgboost"),
    ParamDbl$new("regr.xgboost.colsample_bytree", lower = 0.1, upper = 1, default = 1, tags = "xgboost"),
    ParamDbl$new("regr.xgboost.colsample_bylevel", lower = 0.1, upper = 1, default = 1, tags = "xgboost"),
    ParamDbl$new("regr.xgboost.alpha", lower = 10^-11, upper = 10^-2, default = 10^-11, tags = "xgboost"), # needs trafo
    ParamDbl$new("regr.xgboost.lambda", lower = 10^-11, upper = 10^-2, default = 10^-11, tags = "xgboost"), # needs trafo
    ParamFct$new("regr.xgboost.sample_type", c("uniform", "weighted"), default = "uniform", tags = "xgboost"),
    ParamFct$new("regr.xgboost.normalize_type", c("tree", "forest"), default = "tree", tags = "xgboost"),
    ParamDbl$new("regr.xgboost.rate_drop", lower = 10^-11, upper = 10^0, default = 10^0, tags = "xgboost") # needs trafo
  ))

  ps$add_dep("regr.xgboost.sample_type", "regr.xgboost.booster", CondEqual$new("dart"))
  ps$add_dep("regr.xgboost.normalize_type", "regr.xgboost.booster", CondEqual$new("dart"))
  ps$add_dep("regr.xgboost.rate_drop", "regr.xgboost.booster", CondEqual$new("dart"))

  return(ps)
}

