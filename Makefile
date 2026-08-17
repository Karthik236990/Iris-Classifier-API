.PHONY: install data features train pipeline serve test clean

install:
	pip install -r requirements.txt

data:
	python -m src.data.make_dataset

features:
	python -m src.features.build_features

train:
	python -m src.models.train_model

pipeline: data features train

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.pkl models/*.json __pycache__ .pytest_cache
