# Wraps a `Learner` basically completely, with the exception that a `expire_time` is given (a POSIXct).
# When this time has passed, then this Learner always errors immediately.
# Furthermore, the timeout of the learner is set automatically to the remaining time, should it be less
# than the learner's original timeout.
# The learner should ideally be encapsulated with something that allows hard timeouts, like 'callr' or similar.
# LearnerWrapperExpire inherits this encapsulation and sets the wrapped learner's encapsulation to "none".
# LearnerWrapper furthermore inherits the learner's fallback learner, and sets the wrapped learner's fallback learner to NULL.
# This encapsulates by running wrapped_learner$train() from within private$.train(), and wrapped_learner$predict() from within private$.predict().
LearnerWrapperExpire = R6Class("LearnerWrapperExpire", inherit = mlr3::Learner,
  public = list(
    id = NULL,
    state = NULL,
    task_type = NULL,
    predict_types = NULL,
    feature_types = NULL,
    properties = NULL,
    data_formats = NULL,
    packages = NULL,
    predict_sets = "test",
    fallback = NULL,
    man = NULL,
    expire_time = NULL,
    initialize = function(wrapped_learner, expire_time) {
      # initialize by basically copying everything from wrapped_learner to top level values
      # fallback is set to NULL, because we only want to fall back once.
      # When the user calls $train(), the call goes like this (simplified):
      # LearnerWrapperExpire$train --> mlr3:::learner_train --> ['evaluate' encapsulation] --> LearnerWrapperExpire$private$.train -->
      #   wrapped_learner$train --> mlr3:::learner_train --> [learner's encapsulation] --> wrapped_learner$private$.train
      # similar for predict().
      assert_class(wrapped_learner, "Learner")
      super$initialize(wrapped_learner$id, wrapped_learner$task_type, ParamSet$new(),
        predict_types = wrapped_learner$predict_types, feature_types = wrapped_learner$feature_types,
        properties = wrapped_learner$properties, data_formats = wrapped_learner$data_formats, packages = wrapped_learner$packages,
        man = wrapped_learner$man)
      private$.learner = wrapped_learner$clone(deep = TRUE)
      self$fallback = private$.learner$fallback
      private$.learner$fallback = NULL
      self$encapsulate = map_chr(private$.learner$encapsulate, function(x) if (x == "none") "none" else "evaluate")
      self$expire_time = expire_time
    }
  ),
  active = list(
    hash = function() {
      # change the hash here to avoid confusion between wrapped and unwrapped learners
      digest::digest(list("LearnerWrapperExpire", private$.learner$hash, self$fallback$hash))
    },
    phash = function() {
      # change the hash here to avoid confusion between wrapped and unwrapped learners
      digest::digest(list("LearnerWrapperExpire", private$.learner$phash, self$fallback$hash))
    },
    predict_type = function(rhs) {
      if (!missing(rhs)) {
        private$.learner$predict_type = rhs
      } else {
        private$.learner$predict_type
      }
    },
    param_set = function(rhs) {
      if (!missing(rhs)) {
        private$.learner$param_set = rhs
      } else {
        private$.learner$param_set
      }
    },
    wrapped_learner = function(rhs) {
      if (!missing(rhs)) {
        stop("$learner is purely read-only.")
      }
      learner = private$.learner$clone(deep = TRUE)
      learner$state = self$model
      learner
    }
  ),
  private = list(
    .learner = NULL,
    .train = function(task) {
      timeout = as.numeric(difftime(self$expire_time, Sys.time(), units = "secs"))
      if (timeout < 0) {
        mlr3misc::stopf("Learner %s expired at %s, now it is %s!", self$id, self$expire_time, Sys.time())
      }
      on.exit({private$.learner$state = NULL})
      private$.learner$timeout = pmin(private$.learner$timeout, timeout)
      inloglength = NROW(private$.learner$log)
      private$.learner$train(task)
      outloglength = NROW(private$.learner$log)
      loglines = seq_len(outloglength - inloglength) + inloglength
      # print messages, throw errors etc. of learner.
      for (i in seq_row(private$.learner$log)) {
        curlog = private$.learner$log[i, ]
        switch(as.character(curlog$class), output = message, warning = warning, error = stop)(curlog$msg)
      }
      state = private$.learner$state
      state
    },
    .predict = function(task) {
      timeout = as.numeric(difftime(self$expire_time, Sys.time(), units = "secs"))
      if (timeout <= 0) {
        mlr3misc::stopf("Learner %s expired at %s, now it is %s!", self$id, self$expire_time, Sys.time())
      }
      on.exit({private$.learner$state = NULL})
      private$.learner$timeout = pmin(private$.learner$timeout, timeout)
      private$.learner$state = self$model
      inloglength = NROW(private$.learner$log)
      result = private$.learner$predict(task)
      outloglength = NROW(private$.learner$log)
      loglines = seq_len(outloglength - inloglength) + inloglength
      # print messages, throw errors etc. of learner.
      for (i in seq_row(private$.learner$log)) {
        curlog = private$.learner$log[i, ]
        switch(as.character(curlog$class), output = message, warning = warning, error = stop)(curlog$msg)
      }
      result
    }
  )
)


# Wraps a `Tuner`, but adding a timeout that is passed on to the Learner.
# A terminator is added automatically, adhering to the timeout; additional terminators
# may be present. A TerminatorRunTime is *not* necessary.
# This is most useful when the learner in the TuningInstance is encapsulated with something
# that allows hard timeouts, like 'callr' or similar.
# The learner must have a fallback learner.
TunerWrapperHardTimeout = R6Class("TunerWrapperTimeout", inherit = mlr3tuning::Tuner,
  public = list(
    param_set = NULL,
    param_classes = NULL,
    properties = NULL,
    packages = NULL,
    timeout = NULL,
    initialize = function(tuner, timeout) {
      super$initialize(param_set = tuner$param_set, param_classes = tuner$param_classes,
        properties = tuner$properties, packages = tuner$packages)
      self$timeout = assert_number(timeout, lower = 0)
      private$.tuner = tuner
    },
    optimize = function(inst) {
      assert_multi_class(inst, c("TuningInstanceSingleCrit", "TuningInstanceMultiCrit"))
      if (is.finite(self$timeout)) {
        expiration =  Sys.time() + self$timeout

        learner_orig = inst$objective$learner

        if (is.null(learner_orig$fallback)) {
          stopf("Learner %s must have a fallback learner.", learner_orig$id)
        }

        on.exit({inst$objective$learner = learner_orig})
        terminator_orig = inst$terminator
        on.exit({inst$terminator = terminator_orig})

        tct = bbotk::TerminatorClockTime$new()
        tct$param_set$values$stop_time = expiration
        inst$terminator = bbotk::TerminatorCombo$new(list(tct, terminator_orig))
        inst$objective$learner = LearnerWrapperExpire$new(learner_orig, expiration)
      }
      private$.tuner$optimize(inst)
    }
  ),
  private = list(
    .tuner = NULL
  )
)

OptimizerChain = R6Class("OptimizerChain", inherit = bbotk::Optimizer,
  public = list(
    param_set = NULL,
    param_classes = NULL,
    properties = NULL,
    packages = NULL,
    initialize = function(optimizers, additional_terminators = rep(list(NULL), length(optimizers))) {
      assert_list(optimizers, types = c("Tuner", "Optimizer"), any.missing = FALSE)
      assert_list(additional_terminators, types = c("Terminator", "NULL"), len = length(optimizers))

      param_sets = list()
      ids_taken = character(0)
      for (i_opt in seq_along(optimizers)) {
        opt = optimizers[[i_opt]]
        ps = opt$param_set$clone(deep = TRUE)
        ps$set_id = class(opt)[[1]]
        try_prefix = 0
        while (ps$set_id %in% ids_taken) {
          try_prefix = try_prefix + 1
          ps$set_id = paste0(class(opt)[[1]], "_", try_prefix)
        }
        ids_taken[[i_opt]] = ps$set_id
        param_sets[[i_opt]] = ps
      }
      super$initialize(param_set = ParamSetCollection$new(param_sets),
        param_classes = Reduce(intersect, map(optimizers, "param_classes")),
        properties = Reduce(intersect, map(optimizers, "properties")),
        packages = unique(unlist(map(optimizers, "packages")))
      )
      private$.optimizers = optimizers
      private$.additional_terminators = additional_terminators
    },
    optimize = function(inst) {
      terminator_orig = inst$terminator
      on.exit({inst$terminator = terminator_orig})
      for (i_opt in seq_along(private$.optimizers)) {

        term = private$.additional_terminators[[i_opt]]
        if (!is.null(term)) {
          inst$terminator = bbotk::TerminatorCombo$new(list(term, terminator_orig))
        } else {
          inst$terminator = terminator_orig
        }
        opt = private$.optimizers[[i_opt]]
        opt$param_set$values = self$param_set$.__enclos_env__$private$.sets[[i_opt]]$values
        opt$optimize(inst)
        if (terminator_orig$is_terminated(inst$archive)) break
      }
    }
  ),
  private = list(
    .optimizers = NULL,
    .additional_terminators = NULL
  )
)

TunerChain = R6Class("TunerRandomSearch",
  inherit = mlr3tuning::TunerFromOptimizer,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(...) {
      super$initialize(
        optimizer = OptimizerChain$new(...)
      )
    }
  )
)

