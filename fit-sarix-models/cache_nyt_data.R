library("readr")
library("dplyr")

us_url     <- "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv"
states_url <- "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"

us <- readr::read_csv(us_url,
                      col_types = readr::cols(
                        date   = readr::col_date(format = "%Y-%m-%d"),
                        cases  = readr::col_integer(),
                        deaths = readr::col_integer()
                      )) 

states <- readr::read_csv(states_url,
                          col_types = readr::cols(
                            date   = readr::col_date(format = "%Y-%m-%d"),
                            state  = readr::col_character(),
                            fips   = readr::col_character(),
                            cases  = readr::col_integer(),
                            deaths = readr::col_integer()
                          )) 


readr::write_csv(us,       path = paste0(relative_path,"raw/us.csv"))
readr::write_csv(states,   path = paste0(relative_path,"raw/us-states.csv"))

d <- us %>%
  dplyr::mutate(location = "US") %>%
  dplyr::bind_rows(states %>% 
                     dplyr::rename(location = fips) %>%
                     dplyr::select(-state)
  ) %>%
  dplyr::group_by(location) %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    inc_deaths = as.integer(diff(c(0, deaths))),
    inc_cases  = as.integer(diff(c(0, cases )))) %>%
  dplyr::arrange(location, date) 


location_info <- readr::read_csv('data/locations.csv')
d <- d %>%
  dplyr::left_join(location_info %>% dplyr::transmute(location, pop100k = population / 100000)) %>%
  dplyr::mutate(case_rate = inc_cases / pop100k)

reference_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)
readr::write_csv(d, paste0('data/nyt_data_cached_', reference_date, '.csv'))
