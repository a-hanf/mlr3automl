test_that("basic examples work", {
  test_regression_task = function(task_type, task_id, min_performance, learners = NULL, terminator = NULL, timeout = NULL) {
    task = tsk(task_type, task_id)
    model = AutoML(task, measure = msr("regr.mae"), learner_list = learners, terminator = terminator, learner_timeout = timeout)
    result = model$resample()
    print(paste0("Performance on unseen data: ", result$aggregate(model$measure)))
    expect_lte(result$aggregate(model$measure), min_performance)
    invisible(NULL)
  }

  learners = NULL
  terminator = trm("none")
  timeout = NULL

  test_regression_task("mtcars", "mtcars", 5, learners = learners, terminator = terminator, timeout)
  test_regression_task("boston_housing", "boston_housing", 5, learners = learners, terminator = terminator, timeout)

  test_regression_task("oml", 2295, 50, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 52948, 5, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 4823, 0.75, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 2285, 0.65, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 4729, 200, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 4990, 0.5, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 4958, 0.8, learners = learners, terminator = terminator, timeout)
  test_regression_task("oml", 2280, 0.2, learners = learners, terminator = terminator, timeout)
})
