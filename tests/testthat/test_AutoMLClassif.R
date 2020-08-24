test_that("basic examples work", {
  test_classification_task = function(task_type, task_id, min_performance, learners = NULL) {
    task = tsk(task_type, task_id)
    model = AutoML(task, learner = learners)
    result = model$resample()
    expect_gt(result$aggregate(model$measures), min_performance)
    invisible(NULL)
  }

  test_classification_task("iris", "iris", 0.9)
  test_classification_task("iris", "iris", 0.9, c("classif.ranger", "classif.xgboost"))

  test_classification_task("sonar", "sonar", 0.7)
  test_classification_task("spam", "spam", 0.8)
  test_classification_task("wine", "wine", 0.8)
  test_classification_task("zoo", "zoo", 0.8)

  # taken from https://www.openml.org/search?q=tags.tag%3Astudy_218&type=data&table=1&size=39
  # as referenced in https://openml.github.io/automlbenchmark/benchmark_datasets.html
  test_classification_task("oml", 12L, 0.75)
  test_classification_task("oml", 53L, 0.6)
  test_classification_task("oml", 10101L, 0.7)
  test_classification_task("oml", 9952L, 0.7)
  test_classification_task("oml", 145854L, 0.85)
  test_classification_task("oml", 14965L, 0.85)
  test_classification_task("oml", 9981L, 0.55)
  test_classification_task("oml", 9952L, 0.7)

  # fails due to missing values
  # test_classification_task("pima", "pima", 0.8)
  # fails: Assertion on 'truth' failed: Must have length >= 1, but has length 0.
  # test_classification_task("german_credit", "german_credit", 0.7)
  # fails: R session hangs
  # test_classification_task("oml", 34539L, 0.8)
})

