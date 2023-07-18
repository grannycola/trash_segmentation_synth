install:
	pip install -r requirements.txt;

train:
	cd src/models && python3 train_model.py $(ARGS);

eval:
	python3 src/models/eval_model.py $(ARGS);

mlflow:
	mlflow ui --backend-store-uri src/models/mlruns;
