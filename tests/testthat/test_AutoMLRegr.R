test_that("basic example works", {

  test_standard_task = function(task_type, task_id, min_performance) {
    task = tsk(task_type, task_id)
    model = AutoML(task)
    train_set = sample(model$task$nrow, 0.8 * model$task$nrow)
    test_set = setdiff(seq_len(model$task$nrow), train_set)
    model$train(train_set)
    expect_lt(model$learner$model$tuning_instance$result$perf, min_performance)
    prediction = model$predict(row_ids = test_set)
    expect_lt(prediction$score(model$measures), min_performance)
  }

  test_standard_task("mtcars", "mtcars", 5)
  test_standard_task("boston_housing", "boston_housing", 5)

  # none of these work: Error in open.connection(con, "rb") : HTTP error 412.
  # test_openml_task(2295, 50)
  # test_openml_task(52948, 5)
  # test_openml_task(4823, 0.75)
  # test_openml_task(2285 , 0.75)
  # test_openml_task(4729, 200)
})
