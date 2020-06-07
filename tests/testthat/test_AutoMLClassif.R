test_that("basic example works", {

  test_classification_task = function(task_type, task_id, min_performance) {
    task = tsk(task_type, task_id)
    model = AutoML(task)
    result = model$resample()
    expect_gt(model$learner$model$tuning_instance$result$perf, min_performance)
    expect_gt(result$aggregate(model$measures), min_performance)
    invisible(NULL)
  }

  test_classification_task("iris", "iris", 0.9)
  test_classification_task("german_credit", "german_credit", 0.7)
  # missing data -> error in PipeOp$train: Assertion on 'input 1 ("input") of PipeOp classif.rpart.removeconstants
  # test_classification_task("pima", "pima", 0.8)
  test_classification_task("sonar", "sonar", 0.7)
  test_classification_task("spam", "spam", 0.8)
  test_classification_task("wine", "wine", 0.8)
  test_classification_task("zoo", "zoo", 0.8)

  # taken from https://www.openml.org/search?q=tags.tag%3Astudy_218&type=data&table=1&size=39
  # as referenced in https://openml.github.io/automlbenchmark/benchmark_datasets.html
  test_classification_task("oml", 12, 0.75)
  test_classification_task("oml", 31L, 0.55)
  test_classification_task("oml", 53, 0.6)
  test_classification_task("oml", 10101L, 0.7)
  test_classification_task("oml", 9952L, 0.7)
  test_classification_task("oml", 145854, 0.85)
  test_classification_task("oml", 14965, 0.85)
  test_classification_task("oml", 34539, 0.8)
  test_classification_task("oml", 9981, 0.55)
  test_classification_task("oml", 9952L, 0.7)
})
