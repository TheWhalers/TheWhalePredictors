# This script contains general R code to use for everybody.
# Person-specific code can be stored in your own repository.
# Happy whale spotting!

# Packages ----
library(dplyr)
library(data.table)
library(filesstrings)

# Default directories ----
ids <- fread(file = file.path("data", "train.csv")) %>% as_tibble()
dir_validation <- file.path("data", "validation")
dir.create(dir_validation)
dir_train <- file.path("data", "train")
dir.create(dir_train)

# Total possible classes
nbr_classes <- ids$Id %>% unique() %>% length()

# Define the folder where the zip file should be unzipped to 
output_dir <- file.path("data", "all data") 
dir.create(output_dir)
unzip(file.path("data", "train.zip"), exdir = output_dir) 

# Create a folder for each possible class in both the training and validation folder
for(i in 1:nbr_classes){
  dir.create(file.path("data", "train", unique(ids$Id)[i]))
  dir.create(file.path("data", "validation", unique(ids$Id)[i]))
}

# Determine which files to move - TRAIN
set.seed(100)
# Create list of files with their respective folders to move into
files_to_move <- list.files(output_dir, include.dirs = F, pattern = ".jpg") %>%
  as_tibble() %>%
  sample_frac(size = 0.80) %>% # sample 80% training data
  rename(Image = value) %>%
  left_join( # join with the list of all files to get the correct class
    ids
  ) %>%
  mutate(
    Image = file.path("data", "all data", Image),
    Id = file.path("data", "train", Id)
  )

# Move the files to respective directories
move_files(
  files_to_move %>% pull(Image),
  files_to_move %>% pull(Id)
)
 
# Determine which files to move - VALIDATION
files_to_move <- list.files(output_dir, include.dirs = F, pattern = ".jpg") %>%
  as_tibble() %>%
  rename(Image = value) %>%
  left_join(
    ids
  ) %>%
  mutate(
    Image = file.path("data", "all data", Image),
    Id = file.path("data", "validation", Id)
  )

# Move the files to respective directories
move_files(
  files_to_move %>% pull(Image),
  files_to_move %>% pull(Id)
)
