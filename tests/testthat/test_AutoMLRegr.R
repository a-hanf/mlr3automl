test_that("basic example works", {

  test_regression_task = function(task_type, task_id, min_performance) {
    task = tsk(task_type, task_id)
    model = AutoML(task)
    result = model$resample()
    expect_lt(model$learner$model$tuning_instance$result$perf, min_performance)
    expect_lt(result$aggregate(model$measures), min_performance)
    invisible(NULL)
  }

  test_regression_task("mtcars", "mtcars", 5)
  test_regression_task("boston_housing", "boston_housing", 5)
  test_regression_task("oml", 2295, 50)
  test_regression_task("oml", 52948, 5)
  test_regression_task("oml", 4823, 0.75)
  test_regression_task("oml", 4729, 200)
})
