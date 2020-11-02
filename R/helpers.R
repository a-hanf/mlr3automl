#' @title Create an AutoTuner with a single line of code
#' @name create_autotuner
#' @description
#' Small utility function, which creates an AutoTuner for given learners. The
#' learner in this AutoTuner is a (somewhat complex) GraphLearner used in
#' `mlr3automl`. Will be simplified when preprocessing operations are refactored.
#' @param learner_list [`List of Learners`][mlr3::Learner] \cr
#'   A list of learners
#' @param task [`Task`] \cr
#'   A [`Task`][mlr3::Task] to create the AutoTuner for. Influences the
#'   preprocessing operations in the learner. Will be removed when preprocessing
#'   operations are refactored.
#' @return [`AutoTuner`]
#' @examples
#' \donttest{
#' library(mlr3automl)
#' my_autotuner = create_autotuner(c("classif.svm"))
#' }
#' @export
create_autotuner = function(learner_list = c("classif.ranger"), task = tsk("iris")) {
  model = AutoML(task, learner_list)
  return(model$learner)
}
