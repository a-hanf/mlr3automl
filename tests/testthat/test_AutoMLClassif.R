test_that("basic example works", {

  test_standard_task = function(task_type, task_id, min_performance) {
    task = tsk(task_type, task_id)
    model = AutoML(task)
    train_set = sample(model$task$nrow, 0.8 * model$task$nrow)
    test_set = setdiff(seq_len(model$task$nrow), train_set)
    model$train(train_set)
    expect_gt(model$learner$model$tuning_instance$result$perf, min_performance)
    prediction = model$predict(row_ids = test_set)
    expect_gt(prediction$score(model$measures), min_performance)
  }

  test_standard_task("iris", "iris", 0.9)
  test_standard_task("german_credit", "german_credit", 0.7)
  # missing data -> error in PipeOp$train: Assertion on 'input 1 ("input") of PipeOp classif.rpart.removeconstants
  # test_standard_task("pima", "pima", 0.8)
  test_standard_task("sonar", "sonar", 0.7)
  test_standard_task("spam", "spam", 0.8)
  test_standard_task("wine", "wine", 0.8)
  test_standard_task("zoo", "zoo", 0.8)

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
