# Load required libraries
library(tidymodels)
set.seed(123)

# Load data
data(iris)

# Split into training and testing sets
iris_split <- initial_split(iris, prop = 0.8, strata = Species)
iris_train <- training(iris_split)
iris_test  <- testing(iris_split)

# Create recipe for preprocessing
iris_recipe <- recipe(Species ~ ., data = iris_train) %>%
  step_normalize(all_predictors())   # Normalize numeric inputs for neural nets

# Specify a multilayer perceptron model
mlp_model <- mlp(
  hidden_units = 5,    # number of neurons in the hidden layer
  penalty = 0.01,      # weight decay / regularization term
  epochs = 100         # number of training epochs
) %>%
  set_mode("classification") %>%
  set_engine("nnet")   # you can also use keras if installed

# Bundle model and recipe into a workflow
mlp_workflow <- workflow() %>%
  add_model(mlp_model) %>%
  add_recipe(iris_recipe)

# Fit the model
mlp_fit <- fit(mlp_workflow, data = iris_train)

# Evaluate on test data
mlp_predictions <- predict(mlp_fit, iris_test) %>%
  bind_cols(iris_test)

# Compute accuracy
metrics <- mlp_predictions %>%
  metrics(truth = Species, estimate = .pred_class)

metrics
