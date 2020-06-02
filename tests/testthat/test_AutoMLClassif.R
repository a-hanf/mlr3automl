test_that("basic example works", {
  expect_r6(
    {
      auto_iris <- AutoML(mlr3::tsk("iris"))
    },
    "AutoMLClassif"
  )
  train_set <- sample(auto_iris$task$nrow, 0.8 * auto_iris$task$nrow)
  test_set <- setdiff(seq_len(auto_iris$task$nrow), train_set)
  auto_iris$train(train_set)
  # check train performance
  expect_gt(auto_iris$learner$model$tuning_instance$result$perf, 0.8)
  # check test performance
  prediction <- auto_iris$predict(row_ids = test_set)
  expect_gt(prediction$score(auto_iris$measures), 0.8)

  # # TODO: fix this test, currently AutoML breaks whenever data is missing
  # ind = rnorm(150) > 2
  # iris[ind, "Sepal.Length"] = NA
  # missing_iris = TaskClassif$new("missing_iris", backend = iris, target = "Species")
  # expect_r6({auto_iris = AutoML(missing_iris)}, "AutoMLClassif")
  # train_set = sample(auto_iris$task$nrow, 0.8 * auto_iris$task$nrow)
  # test_set = setdiff(seq_len(auto_iris$task$nrow), train_set)
  # auto_iris$train(train_set)
  # # check train performance
  # expect_gt(auto_iris$learner$model$tuning_instance$result$perf, 0.8)
  # # check test performance
  # prediction = auto_iris$predict(row_ids = test_set)
  # expect_gt(prediction$score(auto_iris$measures), 0.8)

  test_openml_task <- function(task_id, min_performance) {
    task <- tsk("oml", data_id = task_id)
    model <- AutoML(task)
    train_set <- sample(model$task$nrow, 0.8 * model$task$nrow)
    test_set <- setdiff(seq_len(model$task$nrow), train_set)
    model$train(train_set)
    expect_gt(model$learner$model$tuning_instance$result$perf, min_performance)
    prediction <- model$predict(row_ids = test_set)
    expect_gt(prediction$score(model$measures), min_performance)
  }

  # taken from https://www.openml.org/search?q=tags.tag%3Astudy_218&type=data&table=1&size=39
  # as referenced in https://openml.github.io/automlbenchmark/benchmark_datasets.html
  test_openml_task(12, 0.75)
  test_openml_task(31L, 0.55)
  test_openml_task(53, 0.6)

  # these tasks don't run with a 412 response code - offline on OpenML?
  # test_openml_task(10101L, 0.7)
  # test_openml_task(9952L, 0.7)
  # test_openml_task(145854, 0.85)
  # test_openml_task(14965, 0.85)t
  # test_openml_task(34539, 0.8)
  # test_openml_task(9981, 0.55)
  # test_openml_task(9952L, 0.7)
})
