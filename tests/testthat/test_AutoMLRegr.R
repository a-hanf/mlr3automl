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
})
