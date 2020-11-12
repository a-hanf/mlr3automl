set.seed(42)

test_classification_task = function(task_type, task_id, min_performance, learners = NULL, terminator = NULL, timeout, preprocessing = NULL, runtime = Inf) {
  task = tsk(task_type, task_id)
  model = AutoML(task, learner_list = learners, resampling = rsmp("holdout"), terminator = terminator, learner_timeout = timeout, preprocessing = preprocessing, runtime = runtime)
  result = model$resample()
  print(paste0("Performance on unseen data: ", result$aggregate(model$measure)))
  expect_gte(result$aggregate(model$measure), min_performance)
  return(result)
}

learners = NULL
terminator = trm("none")
timeout = 30
preprocessing = "full"
runtime = 120

res = test_classification_task("german_credit", "german_credit", 0.7, learners, terminator, timeout, preprocessing, runtime)
