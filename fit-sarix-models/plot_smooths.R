library(tidyverse)

reference_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)

jhu_data <- readr::read_csv(paste0("data/jhu_data_smoothed_", reference_date, ".csv"))
#nyt_data <- readr::read_csv(paste0("data/nyt_data_smoothed_", reference_date, ".csv"))
cdc_data <- readr::read_csv(paste0("data/cdc_data_smoothed_", reference_date, ".csv"))
# , col_types = cols(
#   location = col_character(),
#   date = col_date(format = ""),
#   case_rate = col_double(),
#   case_rate_sqrt = col_double(),
#   case_rate_fourthrt = col_double(),
#   case_rate_sqrt_trend = col_double(),
#   case_rate_sqrt_seasonal = col_double(),
#   case_rate_fourthrt_trend = col_double(),
#   case_rate_fourthrt_seasonal = col_double()
# ))

pdf(paste0('weekly-submission/sarix-plots/data/data_smoothed_sqrt_', reference_date, '.pdf'), width = 12, height = 8)
for (loc in unique(jhu_data$location)) {
  jhu_loc_data <- jhu_data %>%
    dplyr::filter(location == loc) %>%
    dplyr::select(date, location, case_rate_sqrt, corrected_case_rate_sqrt_taylor_0) %>%
    dplyr::mutate(source = "jhu")
  # nyt_loc_data <- nyt_data %>%
  #   dplyr::filter(location == loc, date >= "2020-10-01") %>%
  #   dplyr::select(date, location, case_rate_sqrt, corrected_case_rate_sqrt_taylor_0) %>%
  #   dplyr::mutate(source = "nyt")
  cdc_loc_data <- cdc_data %>%
    dplyr::filter(location == loc, date >= "2020-10-01") %>%
    dplyr::select(date, location, case_rate_sqrt, corrected_case_rate_sqrt_taylor_0) %>%
    dplyr::mutate(source = "cdc")
  loc_data <- dplyr::bind_rows(jhu_loc_data,
                              #  nyt_loc_data,
                               cdc_loc_data)
  loc_data <- loc_data %>%
    tidyr::pivot_longer(c("case_rate_sqrt", "corrected_case_rate_sqrt_taylor_0")) %>%
    dplyr::mutate(
      signal = dplyr::case_when(
        name == "case_rate_sqrt" ~ "case_rate",
        TRUE ~ "trend"))
  loc_data <- dplyr::left_join(
    loc_data,
    loc_data %>%
      dplyr::filter(signal == "trend") %>%
      dplyr::group_by(date) %>%
      dplyr::summarise(
        combined_value = median(value)
      ),
    by = c("date")
  )
  p <- ggplot(data = loc_data) +
    geom_line(mapping = aes(x = date, y = value, color = signal), size = 1) +
    geom_line(mapping = aes(x = date, y = combined_value)) +
    facet_wrap( ~ source, ncol = 1) +
    ggtitle(loc) +
    theme_bw()
  print(p)
}
dev.off()



pdf(paste0('weekly-submission/sarix-plots/data/data_smoothed_fourthrt_', reference_date, '.pdf'), width = 12, height = 8)
for (loc in unique(jhu_data$location)) {
  jhu_loc_data <- jhu_data %>%
    dplyr::filter(location == loc) %>%
    dplyr::select(date, location, case_rate_fourthrt, corrected_case_rate_fourthrt_taylor_0) %>%
    dplyr::mutate(source = "jhu")
  # nyt_loc_data <- nyt_data %>%
  #   dplyr::filter(location == loc, date >= "2020-10-01") %>%
  #   dplyr::select(date, location, case_rate_fourthrt, corrected_case_rate_fourthrt_taylor_0) %>%
  #   dplyr::mutate(source = "nyt")
  cdc_loc_data <- cdc_data %>%
    dplyr::filter(location == loc, date >= "2020-10-01") %>%
    dplyr::select(date, location, case_rate_fourthrt, corrected_case_rate_fourthrt_taylor_0) %>%
    dplyr::mutate(source = "cdc")
  loc_data <- dplyr::bind_rows(jhu_loc_data,
                              #  nyt_loc_data,
                               cdc_loc_data)
  loc_data <- loc_data %>%
    tidyr::pivot_longer(c("case_rate_fourthrt", "corrected_case_rate_fourthrt_taylor_0")) %>%
    dplyr::mutate(
      signal = dplyr::case_when(
        name == "case_rate_fourthrt" ~ "case_rate",
        TRUE ~ "trend"))
  loc_data <- dplyr::left_join(
    loc_data,
    loc_data %>%
      dplyr::filter(signal == "trend") %>%
      dplyr::group_by(date) %>%
      dplyr::summarise(
        combined_value = median(value)
      ),
    by = c("date")
  )
  p <- ggplot(data = loc_data) +
    geom_line(mapping = aes(x = date, y = value, color = signal), size = 1) +
    geom_line(mapping = aes(x = date, y = combined_value)) +
    facet_wrap( ~ source, ncol = 1) +
    ggtitle(loc) +
    theme_bw()
  print(p)
}
dev.off()



# pdf('weekly-submission/sarix-plots/data/data_smoothed_fourthrt_2022-04-18.pdf', width = 12, height = 8)
# for (loc in unique(data$location)) {
#   loc_data <- data %>% dplyr::filter(location == loc)
#   p <- ggplot(data = loc_data) +
#     geom_line(mapping = aes(x = date, y = case_rate_fourthrt)) +
#     geom_line(mapping = aes(x = date, y = case_rate_fourthrt_trend), color = "orange") +
#     ggtitle(loc) +
#     theme_bw()
#   print(p)
# }
# dev.off()

