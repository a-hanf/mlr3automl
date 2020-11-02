test_that("performance is reasonable", {
  test_classification_task = function(task_type, task_id, min_performance, learners = NULL, terminator = NULL, timeout = NULL) {
    task = tsk(task_type, task_id)
    model = AutoML(task, learner_list = learners, resampling = rsmp("holdout"), terminator = terminator, learner_timeout = timeout)
    result = model$resample()
    print(paste0("Performance on unseen data: ", result$aggregate(model$measure)))
    expect_gte(result$aggregate(model$measure), min_performance)
    return(result)
  }

  learners = NULL
  terminator = trm("none")
  timeout = NULL

  res = test_classification_task("german_credit", "german_credit", 0.7, learners, terminator, timeout)
  res = test_classification_task("oml", 146821L, 0.7, learners, terminator, timeout)

  res = test_classification_task("iris", "iris", 0.9, learners, terminator, timeout)

  res = test_classification_task("sonar", "sonar", 0.7, learners, terminator, timeout)
  res = test_classification_task("spam", "spam", 0.8, learners, terminator, timeout)
  res = test_classification_task("wine", "wine", 0.8, learners, terminator, timeout)
  res = test_classification_task("zoo", "zoo", 0.8, learners, terminator, timeout)

  # taken from https://www.openml.org/search?q=tags.tag%3Astudy_218&type=data&table=1&size=39
  # as referenced in https://openml.github.io/automlbenchmark/benchmark_datasets.html
  res = test_classification_task("oml", 12L, 0.75, learners, terminator, timeout)

  res = test_classification_task("oml", 53L, 0.6, learners, terminator, timeout)
  res = test_classification_task("oml", 10101L, 0.7, learners, terminator, timeout)
  res = test_classification_task("oml", 9952L, 0.7, learners, terminator, timeout)
  res = test_classification_task("oml", 9981L, 0.55, learners, terminator, timeout)
  res = test_classification_task("pima", "pima", 0.7, learners, terminator, timeout)

  # datasets too big for my local machine
  # res = test_classification_task("oml", 145854L, 0.85, learners, terminator, timeout)
  # res = test_classification_task("oml", 14965L, 0.85, learners, terminator, timeout)
  # res = test_classification_task("oml", 34539L, 0.96, learners, terminator, timeout)
})

