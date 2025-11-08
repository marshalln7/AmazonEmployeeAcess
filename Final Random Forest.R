
library(tidyverse)
library(tidymodels)
library(vroom)
library(glue)
library(glmnet)
library(rpart)
library(ranger)
library(embed)
library(themis)

train_data <- vroom("train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test_data <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_bayes(all_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_upsample(ACTION)

my_mod <- rand_forest(mtry = tune(), min_n=tune(), trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

grid_of_tuning_params <- grid_space_filling(mtry(range=c(1, 5)), min_n(), size = 10)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(accuracy, roc_auc))

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

tree_predictions <- final_wf %>%
  predict(new_data = test_data, type="prob")

## Format the Predictions for Submission to Kaggle
kaggle_submission <- tree_predictions %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1) 

vroom_write(x=kaggle_submission, file="./Random_Forest_Final_Predictions.csv", delim=",")

print("All done!")
