library(tidyverse)

base_path <- "weekly-submission/sarix-forecasts"
forecast_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)

for (target_var in c("cases", "hosps")) {
  target_by_loc_path <- file.path(base_path, paste0(target_var, "-by-loc"))
  models <- list.dirs(
    target_by_loc_path,
    full.names = FALSE,
    recursive = FALSE)
#   models <- c("SARIX_covariates_none_smooth_False_transform_fourthrt_p_7_d_0_P_0_D_0",
#               "SARIX_covariates_none_smooth_False_transform_fourthrt_p_14_d_0_P_0_D_0")
#   models <- c("SARIX_covariates_cases_smooth_False_transform_fourthrt_p_7_d_0_P_0_D_0",
#               "SARIX_covariates_cases_smooth_False_transform_fourthrt_p_14_d_0_P_0_D_0")

  for (model in models) {
    files <- list.files(file.path(target_by_loc_path, model))
    date_files <- files[grep(forecast_date, files)]
    date_forecasts <- purrr::map_dfr(
      date_files,
      function(date_file) {
        readr::read_csv(
          file.path(target_by_loc_path, model, date_file),
          col_types = cols(
            location = col_character(),
            forecast_date = col_character(),
            target_end_date = col_character(),
            target = col_character(),
            type = col_character(),
            quantile = col_double(),
            value = col_double()
          )
        )
      }
    )
    save_dir <- file.path(base_path, target_var, model)
    if (!dir.exists(save_dir)) {
      dir.create(save_dir, recursive = TRUE)
    }
    if (!file.exists(file.path(save_dir, paste0(forecast_date, "-", model, ".csv")))) {
      readr::write_csv(
        date_forecasts,
        file.path(save_dir, paste0(forecast_date, "-", model, ".csv"))
      )
    }
  }
}
