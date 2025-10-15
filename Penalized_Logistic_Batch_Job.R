# Run this once interactively
#dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
#install.packages(c("tidymodels", "tidyverse", "vroom", "dplyr", "embed", "glmnet"),
#                 lib = Sys.getenv("R_LIBS_USER"))

library(tidymodels)
library(tidyverse)
library(vroom)
library(dplyr)
library(embed)

train_data <- vroom("train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test_data <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data=train_data) |>
  step_mutate_at(all_predictors(), fn = as.factor) |>
  step_novel(all_nominal_predictors()) |>     # EVALUATE THIS APPROACH LATER!!!
  step_other(all_nominal_predictors(), threshold = 0.002, other="Other") |> #Look at setting this back to threshold = 0.001
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

#prepped_recipe <- prep(my_recipe)
#train_data_2 <- bake(prepped_recipe, new_data=train_data)
#vroom_write(x=train_data_2, file="./Baked Train Data.csv", delim=",")

logRegModel <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

my_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(logRegModel)

## Grid of values to tune over10
tuning_grid <- grid_regular(penalty(), mixture(), levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(train_data, v = 5, repeats=1)

## Run the CV18
CV_results <- my_workflow %>%
  tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  my_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## Predict11
amazon_predictions <- final_wf %>%
  predict(new_data = test_data, type="prob")

## Format the Predictions for Submission to Kaggle
kaggle_submission <- amazon_predictions %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1) 

vroom_write(x=kaggle_submission, file="./Batch_Logistic_Regression_Predictions.csv", delim=",")

