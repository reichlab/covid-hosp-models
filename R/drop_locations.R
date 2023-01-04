# Rscript --vanilla drop_locations.R ID IA MO NY

library(tidyverse)
library(readr)

states_to_drop <- commandArgs(trailingOnly = TRUE)
print(states_to_drop)

locations_data <- read.csv("data-locations/locations.csv")

# double-check the location
location_name <- locations_data %>%
        dplyr::filter(abbreviation %in% states_to_drop) %>%
        dplyr::pull(location_name)
print(location_name)

# get the location code
location_code <- locations_data %>%
        dplyr::filter(abbreviation %in% states_to_drop) %>%
        dplyr::pull(location)
print(location_code)

# read the latest file
file_names <- Sys.glob("data-processed/UMass-trends_ensemble/*.csv")
latest_file <- max(file_names)
print(latest_file)

# copy file for back-up
file.copy(
        latest_file,
        paste0(substr(latest_file, 1, nchar(latest_file) - 4), "-backup.csv")
)


# update the latest file by removing the above locations
updated_file <- read_csv(latest_file) %>%
        filter(!(location %in% location_code))

write_csv(updated_file, latest_file)
