.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: ## runs analysis
	make report/diabetes_analysis.html

.PHONY: cl
cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p osx-arm64

.PHONY: env
env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove 522-project || true
	conda-lock install -n 522-project conda-lock.yml

.PHONY: build
build: ## build the docker image from the Dockerfile
	docker build -t 522-project --file Dockerfile .

.PHONY: run
run: ## alias for the up target
	make up

.PHONY: up
up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker-compose up -d

.PHONY: stop
stop: ## stop docker-compose services
	docker-compose stop

# docker multi architecture build rules (from Claude) -----

.PHONY: docker-build-push
docker-build-push: ## Build and push multi-arch image to Docker Hub (amd64 + arm64)
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag amanbinepal/docker-522-project:latest \
		--tag amanbinepal/docker-522-project:local-$(shell git rev-parse --short HEAD) \
		--push \
		.

.PHONY: docker-build-local
docker-build-local: ## Build single-arch image for local testing (current platform only)
	docker build \
		--tag amanbinepal/docker-522-project:local \
		.

.PHONY: pipeline fetch validate

# pipeline: fetch validate eda model ## Run full pipeline

# fetch: ## Run scripts for fetching and splitting data
# 	python src/01_fetch_data.py
# 	python src/02_split_data.py
# 	python src/03_train_set.py
# 	python src/04_test_set.py

# validate: # Run scripts for data validation
# 	python src/05_data_validation1_1.py
# 	python src/06_data_validation1_2.py
# 	python src/07_data_validation2.py
# 	python src/08_data_validation3.py
# 	python src/09_data_validation4.py
# 	python src/10_data_validation5.py
# 	python src/11_data_validation6.py
# 	python src/12_data_validation7.py
# 	python src/13_data_validation8.py
# 	python src/14_data_validation9.py
# 	python src/15_data_validation10.py

# eda: ## Run EDA
# 	python src/16_eda_visualization.py

# model: ## Run model training and testing
# 	python src/17_model_validation_training.py
# 	python src/18_model_testing.py

.PHONY: clean
clean: ## Clean all generate outputs
	rm -f data/raw/*
	rm -f src/objects/*
	rm -f results/figures/*
	rm -f results/models/*
	rm -f report/diabetes_analysis.html
	rm -rf report/diabetes_analysis_files

# ================================== Dependency Track of 01_fetch_data.py's output
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv: src/01_fetch_data.py
	python src/01_fetch_data.py

# ================================== Dependency Track of 02_split_data.py's output
src/objects/train_df.csv: src/02_split_data.py data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
	python src/02_split_data.py

src/objects/test_df.csv: src/02_split_data.py data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
	python src/02_split_data.py

# ================================== Dependency Track of 03_train_set.py's output
src/objects/X_train.csv: src/03_train_set.py src/objects/train_df.csv
	python src/03_train_set.py

src/objects/y_train.csv: src/03_train_set.py src/objects/train_df.csv
	python src/03_train_set.py

# ================================== Dependency Track of 04_test_set.py's output
src/objects/X_test.csv: src/04_test_set.py src/objects/test_df.csv
	python src/04_test_set.py

src/objects/y_test.csv: src/04_test_set.py src/objects/test_df.csv
	python src/04_test_set.py

# ================================== Dependency Track of 16_eda_visualization.py's output
results/figures/plot_class.png: src/16_eda_visualization.py src/objects/train_df.csv
	python src/16_eda_visualization.py

results/figures/plot_numeric_box.png: src/16_eda_visualization.py src/objects/train_df.csv
	python src/16_eda_visualization.py

results/figures/plot_binary_bar.png: src/16_eda_visualization.py src/objects/train_df.csv
	python src/16_eda_visualization.py

results/figures/plot_ordinal_bar.png: src/16_eda_visualization.py src/objects/train_df.csv
	python src/16_eda_visualization.py

# ================================== Dependency Track of 17_model_validation_training.py's output
results/models/model_training.csv: src/17_model_validation_training.py src/objects/X_train.csv src/objects/y_train.csv
	python src/17_model_validation_training.py

# ================================== Dependency Track of 18_model_testing.py's output
results/models/model_testing.csv: src/18_model_testing.py src/objects/X_test.csv src/objects/y_test.csv
	python src/18_model_testing.py

# ================================== Dependency Track of  output
report/diabetes_analysis.html: report/diabetes_analysis.qmd report/reference.bib src/objects/train_df.csv src/objects/test_df.csv results/models/model_training.csv results/models/model_testing.csv results/figures/plot_class.png results/figures/plot_ordinal_bar.png results/figures/plot_numeric_box.png results/figures/plot_binary_bar.png
	quarto render report/diabetes_analysis.qmd
