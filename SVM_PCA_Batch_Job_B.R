
library(tidymodels)
library(tidyverse)
library(vroom)
library(dplyr)
library(embed)
library(discrim)
library(naivebayes)
library(themis)


train_data <- vroom("train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test_data <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_bayes(all_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_downsample(ACTION)

#prepped_recipe <- prep(my_recipe)
#train_data_2 <- bake(prepped_recipe, new_data=train_data)
#vroom_write(x=train_data_2, file="./Baked Amazon Train Data.csv", delim=",")

svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

my_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(svm_model) |>
  fit(data = train_data)

## Predict11
amazon_predictions <- my_workflow %>%
  predict(new_data = test_data, type="prob")

## Format the Predictions for Submission to Kaggle
kaggle_submission <- amazon_predictions %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1) 

vroom_write(x=kaggle_submission, file="./SVM_RBF_Predictions.csv", delim=",")
