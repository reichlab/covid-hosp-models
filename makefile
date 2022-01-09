all: covidData trends-ensemble

covidData:
	./refresh-covidData.sh

trends-ensemble:
	Rscript R/baseline.R
