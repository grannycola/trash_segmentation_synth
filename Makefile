install:
	pip install -r requirements.txt

train:
	python src/models/train_model.py $(ARGS)

eval:
	python src/models/eval_model.py $(ARGS)