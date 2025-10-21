
library(tidymodels)
library(tidyverse)
library(vroom)
library(dplyr)


train_data <- vroom("train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test_data <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data=train_data) |>
  step_mutate_at(all_predictors(), fn = as.factor) |>
  step_novel(all_nominal_predictors()) |>     # EVALUATE THIS APPROACH LATER!!!
  step_other(all_nominal_predictors(), threshold = 0.004, other="Other") |> #Look at setting this back to threshold = 0.001
  step_dummy(all_nominal_predictors(), one_hot = TRUE)  # Full dummy encoding

#prepped_recipe <- prep(my_recipe)
#train_data_2 <- bake(prepped_recipe, new_data=train_data)
#vroom_write(x=train_data_2, file="./Baked Train Data.csv", delim=",")

knn_model <- nearest_neighbor(neighbors=5) %>%
  set_mode("classification") %>%
  set_engine("kknn")

workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model) |>
  fit(data = train_data)

## Predict7
tree_predictions <- workflow %>%
  predict(new_data = test_data, type = "prob")


## Format the Predictions for Submission to Kaggle
kaggle_submission <- amazon_predictions %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1) 

vroom_write(x=kaggle_submission, file="./KNN_Predictions.csv", delim=",")
```