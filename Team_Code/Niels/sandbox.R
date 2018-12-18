# Packages ----
library(keras)
library(dplyr)
library(data.table)
library(filesstrings)

# MODELLING ----
# Simple Model ----
# Create data generators
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  dir_train,
  train_datagen,
  target_size = c(250, 250),
  batch_size = 100,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  dir_validation,
  validation_datagen,
  target_size = c(250, 250),
  batch_size = 100,
  class_mode = "categorical"
)


# Build own CNN
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(250, 250, 3)) %>%
  layer_batch_normalization(axis = 3) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = 1, activation = "relu") %>%
  layer_average_pooling_2d(pool_size = c(3, 3)) %>%
  # layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  # layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  # layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  # layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = nbr_classes, activation = "softmax")

# create custom metric: top 3 accuracy
metric_top_3_categorical_accuracy <- custom_metric("top_3_categorical_accuracy", function(y_true, y_pred) {
    metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
  })

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam", #optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy", metric_top_3_categorical_accuracy)
)


# Fit model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 20,
  epochs = 5,
  validation_data = validation_generator,
  validation_steps = 20
)

# Save model
model %>% save_model_hdf5("whale_classification_2.h5")

# Load model
model <- load_model_hdf5("whale_classification_2.h5", custom_objects = c("top_3_categorical_accuracy" = metric_top_3_categorical_accuracy))

prediction <- model %>% predict_generator(validation_generator, steps = 100, workers = 4)







# With Data Augmentation ----
train_datagen <- image_data_generator(
  rescale = 1/255,  
  rotation_range = 40,  
  width_shift_range = 0.2,  
  height_shift_range = 0.2,  
  shear_range = 0.2,  
  zoom_range = 0.2,  
  horizontal_flip = TRUE,  
  fill_mode = "nearest"
)

train_generator <- flow_images_from_directory(
  dir_train,
  train_datagen,
  target_size = c(250, 250),
  batch_size = 20,
  class_mode = "categorical"
)

validation_datagen <- image_data_generator(rescale = 1/255)

validation_generator <- flow_images_from_directory(
  dir_validation,
  validation_datagen,
  target_size = c(250, 250),
  batch_size = 20,
  class_mode = "categorical"
)


# Build own CNN
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(10, 10), activation = "relu",
                input_shape = c(250, 250, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 1024, kernel_size = c(10, 10), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = nbr_classes, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("top_k_categorical_accuracy")
)

# Fit model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 50
)

# Save model
model %>% save_model_hdf5("whale_classification_2.h5")









## XCeption Base ----
conv_base <- application_inception_v3(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(250, 250, 3)
)


# add our custom layers
predictions <- conv_base$output %>%
  layer_conv_2d(filters = 1024, kernel_size = c(5, 5), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 2500, activation = "relu") %>%
  layer_dropout(rate = 0.20) %>%
  layer_dense(units = nbr_classes, activation = "softmax")

# this is the model we will train
model <- keras_model(inputs = conv_base$input, outputs = predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
freeze_weights(conv_base)

# compile the model (should be done *after* setting layers to non-trainable)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("top_k_categorical_accuracy")
)


# Create data generators
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  dir_train,
  train_datagen,
  target_size = c(250, 250),
  batch_size = 20,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  dir_validation,
  validation_datagen,
  target_size = c(250, 250),
  batch_size = 20,
  class_mode = "categorical"
)

# Fit model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 50
)
