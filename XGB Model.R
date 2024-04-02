library(tidyverse)
library(caret)
library(xgboost)
library(pROC)
library(ggplot2)
library(dplyr)
library(lubridate)
library(GGally)
library(corrplot)

# Load datasets
train <- read.csv('train.csv')
test <- read.csv('test.csv')

# Initial dataset dimensions
dimensions <- dim(train)
print(paste("The dataset has", dimensions[1], "rows and", dimensions[2], "columns."))

# Summary statistics
summary(train)

# Check for missing values
total_missing_values <- sum(is.na(train))
print(paste("Total missing values in the dataset:", total_missing_values))

# Function to find outliers
find_outliers <- function(x) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = TRUE)
  iqr <- IQR(x, na.rm = TRUE)
  lower <- qnt[1] - 1.5*iqr
  upper <- qnt[2] + 1.5*iqr
  return(sum(x < lower | x > upper, na.rm = TRUE))
}

# Apply the outlier detection function
outliers_count <- sapply(train[, sapply(train, is.numeric)], find_outliers)
print("Outliers count by column:")
print(outliers_count)

# Analyze target variable distribution
target_column <- "churn" # Adjust based on your dataset
class_distribution <- table(train[[target_column]])
print("Class Distribution:")
print(class_distribution)

# Visualize churn rate
churn_rate <- train %>%
  group_by(churn) %>%
  summarise(Count = n()) %>%
  mutate(Churn_Rate = Count / sum(Count))

ggplot(churn_rate, aes(x = as.factor(churn), y = Churn_Rate, fill = as.factor(churn))) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +
  labs(title = "Churn Rate", x = "Churn Status", y = "Proportion") +
  theme_minimal()

# Handle missing values
train <- na.omit(train)
test <- na.omit(test)

# Splitting the data into training and validation sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(train$churn, p = 0.8, list = FALSE)
train_set <- train[train_index, ]
val_set <- train[-train_index, ]

# Preparing matrices for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_set[,-ncol(train_set)]), label = train_set$churn)
dval <- xgb.DMatrix(data = as.matrix(val_set[,-ncol(val_set)]), label = val_set$churn)

# Hyperparameter Tuning (simplified example)
params_grid <- expand.grid(
  eta = c(0.01, 0.1),
  max_depth = c(4, 6, 8),
  subsample = c(0.7, 0.8),
  colsample_bytree = c(0.7, 0.8),
  min_child_weight = c(1, 5),
  gamma = c(0, 0.1)
)

best_auc <- 0
best_params <- NULL
best_nround <- NULL

for(i in 1:nrow(params_grid)) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = params_grid$eta[i],
    gamma = params_grid$gamma[i],
    max_depth = params_grid$max_depth[i],
    subsample = params_grid$subsample[i],
    colsample_bytree = params_grid$colsample_bytree[i],
    min_child_weight = params_grid$min_child_weight[i],
    eval_metric = "auc"
  )
  
  cv.model <- xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = TRUE, stratified = TRUE, print.every.n = 10, early_stopping_rounds = 10, maximize = TRUE)
  
  if (cv.model$evaluation_log$test_auc_mean[cv.model$best_iteration] > best_auc) {
    best_auc <- cv.model$evaluation_log$test_auc_mean[cv.model$best_iteration]
    best_params <- params
    best_nround <- cv.model$best_iteration
  }
}

# Train the model with the best parameters
model_fit <- xgb.train(params = best_params, data = dtrain, nrounds = best_nround)

# Making predictions on the validation set
val_predictions <- predict(model_fit, newdata = dval)
val_labels <- val_set$churn
auc_val <- roc(response = val_labels, predictor = val_predictions)$auc
print(paste("Validation AUC:", auc_val))

# Preparing the test set for predictions using model.matrix
test_matrix <- model.matrix(~ . -1, data = test)
dtest <- xgb.DMatrix(data = test_matrix)

# Generate Predictions
test_preds <- predict(model_fit, dtest)

# Prepare Submission Data Frame
submission <- data.frame(id = test$id, churn = test_preds)
write.csv(submission, 'xgb_submission_7.csv', row.names = FALSE)

print("Model training and prediction process completed. Check 'xgb_submission_7.csv' for the output.")
