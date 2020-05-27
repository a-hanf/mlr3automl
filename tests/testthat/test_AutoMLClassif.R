test_that("basic example works", {
  expect_r6({auto_iris = AutoML(mlr3::tsk("iris"))}, "AutoMLClassif")
  auto_iris$train()
  expect_gt(auto_iris$train_performance, 0.8)
})
