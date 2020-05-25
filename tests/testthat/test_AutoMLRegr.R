library(testthat)

test_that("basic example works", {
  expect_r6(AutoMLRegr$new(mlr3::tsk("mtcars")), "AutoMLRegr")
})
