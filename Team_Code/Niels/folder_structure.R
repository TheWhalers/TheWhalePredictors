# Packages
library(dplyr)
library(data.table)
library(filesstrings)

# Default directories ----
ids <- fread(file = file.path("data", "train.csv")) %>% as_tibble()
dir_validation <- file.path("data", "validation")
dir.create(dir_validation)
dir_train <- file.path("data", "train")
dir.create(dir_train)

# Folder Structure ----
nbr_classes <- ids$Id %>% unique() %>% length()

# Define the folder where the zip file should be unzipped to 
output_dir <- file.path("data", "all data") 
dir.create(output_dir)
unzip(file.path("data", "train.zip"), exdir = output_dir) 


for(i in 1:nbr_classes){
  dir.create(file.path("data", "train", unique(ids$Id)[i]))
  dir.create(file.path("data", "validation", unique(ids$Id)[i]))
}

# Determine which files to move - TRAIN
set.seed(100)
files_to_move <- list.files(output_dir, include.dirs = F, pattern = ".jpg") %>%
  as_tibble() %>%
  sample_frac(size = 0.80) %>% # sample 80% training data
  rename(Image = value) %>%
  left_join(
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
