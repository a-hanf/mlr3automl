library(testthat)

test_that("basic example works", {
  expect_r6(AutoMLClassif$new(mlr3::tsk("iris")), "AutoMLClassif")
})
