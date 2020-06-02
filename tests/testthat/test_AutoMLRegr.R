test_that("basic example works", {
  expect_r6({auto_mtcars = AutoML(mlr3::tsk("mtcars"))}, "AutoMLRegr")
  train_set = sample(auto_mtcars$task$nrow, 0.8 * auto_mtcars$task$nrow)
  test_set = setdiff(seq_len(auto_mtcars$task$nrow), train_set)
  auto_mtcars$train(train_set)
  # check train performance
  expect_lt(auto_mtcars$learner$model$tuning_instance$result$perf, 5)
  # check test performance
  prediction = auto_mtcars$predict(row_ids = test_set)
  expect_lt(prediction$score(auto_mtcars$measures), 5)

  test_openml_task = function(task_id, min_performance) {
    task = tsk("oml", data_id = task_id)
    model = AutoML(task)
    train_set = sample(model$task$nrow, 0.8 * model$task$nrow)
    test_set = setdiff(seq_len(model$task$nrow), train_set)
    model$train(train_set)
    expect_gt(model$learner$model$tuning_instance$result$perf, min_performance)
    prediction = model$predict(row_ids = test_set)
    expect_gt(prediction$score(model$measures), min_performance)
  }

  # none of these work: Error in open.connection(con, "rb") : HTTP error 412.
  # test_openml_task(2295, 50)
  # test_openml_task(52948, 5)
  # test_openml_task(4823, 0.75)
  # test_openml_task(2285 , 0.75)
  # test_openml_task(4729, 200)
})
