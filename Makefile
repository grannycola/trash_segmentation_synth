install:
	pip install -r requirements.txt;

train:
	cd src/cli && python3 train_cli_commands.py $(ARGS);

eval:
	cd src/models && python3 eval_model.py $(ARGS);

mlflow:
	mlflow ui --backend-store-uri reports/mlruns;
