# current date
TODAY_DATE = $(shell date +'%Y-%m-%d')

all: covidData trends-ensemble

covidData:
	./refresh-covidData.sh

trends-ensemble:
	Rscript R/baseline.R

sarix: cache-data transform-data fit-sarix-components combine-by-location build-sarix-ensemble plot-sarix

cache-data:
	Rscript fit-sarix-models/cache_cdc_data.R
	Rscript fit-sarix-models/cache_data.R

transform-data:
	python3 fit-sarix-models/transform_data.py --forecast_date $(TODAY_DATE) --source jhu
	python3 fit-sarix-models/transform_data.py --forecast_date $(TODAY_DATE) --source cdc

fit-sarix-components:
	python3 fit-sarix-models/run_all_models_one_date_by_loc.py --forecast_date $(TODAY_DATE)

combine-by-location:
	Rscript fit-sarix-models/combine_forecasts_by_location.R

build-sarix-ensemble:
	Rscript fit-sarix-models/build_sarix_ensemble.R

plot-sarix:
	Rscript fit-sarix-models/plot_sarix_forecasts.R

