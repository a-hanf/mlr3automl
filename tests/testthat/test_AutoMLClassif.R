test_that("basic example works", {
  expect_r6({auto_iris = AutoML(mlr3::tsk("iris"))}, "AutoMLClassif")
  train_set = sample(auto_iris$task$nrow, 0.8 * auto_iris$task$nrow)
  test_set = setdiff(seq_len(auto_iris$task$nrow), train_set)
  auto_iris$train(train_set)
  # check train performance
  expect_gt(auto_iris$learner$model$tuning_instance$result$perf, 0.8)
  # check test performance
  prediction = auto_iris$predict(row_ids = test_set)
  expect_gt(prediction$score(auto_iris$measures), 0.8)
})
