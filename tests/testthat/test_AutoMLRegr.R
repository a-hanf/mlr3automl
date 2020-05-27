test_that("basic example works", {
  expect_r6({auto_mtcars = AutoML(mlr3::tsk("mtcars"))}, "AutoMLRegr")
  auto_mtcars$train()
  expect_lt(auto_mtcars$train_performance, 5)
})
