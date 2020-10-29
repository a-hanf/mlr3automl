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
AutoML = function(task, learner_list = NULL, learner_timeout = NULL,
                  resampling = NULL, measures = NULL, terminator = NULL) {
  if (task$task_type == "classif") {
    # stratify target variable so that every target label appears
    # in all folds while resampling
    target_is_factor = task$col_info[task$col_info$id == task$target_names, ]$type == "factor"
    if (length(target_is_factor) == 1 && target_is_factor) {
      task$col_roles$stratum = task$target_names
    }
    return(AutoMLClassif$new(task, learner_list, learner_timeout,
                             resampling, measures, terminator))
  } else if (task$task_type == "regr") {
    return(AutoMLRegr$new(task, learner_list, learner_timeout,
                          resampling, measures, terminator))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
