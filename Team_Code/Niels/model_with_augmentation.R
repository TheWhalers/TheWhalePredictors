# Packages ----
library(keras)
library(dplyr)
library(tidyr)
library(data.table)
library(filesstrings)
library(tibble)
library(magrittr)

# Default directories ----
ids <- fread(file = file.path("D:/OneDrive - Acumen Consulting BVBA/R Projects/Whale Classification/data", "train.csv")) %>% as_tibble()
dir_validation <- file.path("D:/OneDrive - Acumen Consulting BVBA/R Projects/Whale Classification/data", "validation")
dir_train <- file.path("D:/OneDrive - Acumen Consulting BVBA/R Projects/Whale Classification/data", "train")

# Folder Structure ----
nbr_classes <- ids$Id %>% unique() %>% length()

# MODELLING ----
# Simple Model ----
# Create data generators
# With Data Augmentation ----
train_datagen <- image_data_generator(
  rescale = 1/255,  
  rotation_range = 40,  
  # width_shift_range = 0.2,  
  # height_shift_range = 0.2,  
  shear_range = 0.2,  
  zoom_range = 0.2,  
  horizontal_flip = TRUE,  
  fill_mode = "nearest"
)

validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  dir_train,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 300,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  dir_validation,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 100,
  class_mode = "categorical"
)


# Build own CNN
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  #layer_batch_normalization(axis = 3) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_average_pooling_2d(pool_size = c(3, 3)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu") %>%
  layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = nbr_classes, activation = "softmax")

# # create custom metric: top 3 accuracy
# metric_top_3_categorical_accuracy <- custom_metric("top_3_categorical_accuracy", function(y_true, y_pred) {
#   metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
# })

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy", "top_k_categorical_accuracy")
)

# Fit model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 30,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 20
)

# Save model
model %>% save_model_hdf5("whale_classification_with_augmentation_v2.h5")

# Load model
model <- load_model_hdf5("whale_classification_with_augmentation.h5")

# Create Predictions
validation_test_generator <- flow_images_from_directory(
  dir_validation,
  validation_datagen,
  target_size = c(250, 250),
  batch_size = 1, # easier to later select how many predictions we want to make
  class_mode = "categorical"
)

nbr_validation_files <- list.files("data/validation", recursive = T, include.dirs = T, pattern = ".jpg") %>% length()

prediction <- model %>% predict_generator(validation_test_generator, steps = nbr_validation_files, workers = 4, verbose = 1)


prediction %>% as_tibble() %>% rowwise() %>% select_if(., is.numeric) %>% rowSums() %>% as_tibble() %>% filter(value > 1)

prediction_with_actuals <- prediction %>%
  as_tibble() %>%
  magrittr::set_colnames(list.files("data/validation")) %>%
  cbind(list.files("data/validation", recursive = T, include.dirs = T, pattern = ".jpg")[1:nbr_validation_files] %>% as_tibble()) %>% 
  rename(img_dir = value) %>% 
  separate(img_dir, into = c("actual", "image"), sep = "/")
