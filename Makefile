install:
	pip install -r requirements.txt;

train:
	cd src/models && python train_model.py $(ARGS);

eval:
	python src/models/eval_model.py $(ARGS);

mlflow:
	mlflow ui --backend-store-uri src/models/mlruns;
