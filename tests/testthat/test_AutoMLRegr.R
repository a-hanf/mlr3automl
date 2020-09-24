test_that("basic examples work", {
  test_regression_task = function(task_type, task_id, min_performance, learners = NULL) {
    task = tsk(task_type, task_id)
    model = AutoML(task, learner_list = learners)
    result = model$resample()
    expect_lte(result$aggregate(model$measures), min_performance)
    invisible(NULL)
  }

  # TODO: write tests for input checking

  test_regression_task("mtcars", "mtcars", 5)
  test_regression_task("mtcars", "mtcars", 5, learners = c('regr.xgboost', 'regr.ranger'))

  test_regression_task("boston_housing", "boston_housing", 5)


  test_regression_task("oml", 2295, 50)
  test_regression_task("oml", 52948, 5)
  test_regression_task("oml", 4823, 0.75)
  test_regression_task("oml", 2285, 0.65)
  test_regression_task("oml", 4729, 200)

  test_regression_task("oml", 4990, 0.5)
  test_regression_task("oml", 4958, 0.8)
  test_regression_task("oml", 2280, 0.2)



})
