PYTHON ?= python3

.PHONY: all notebook check check-pretrained
all: notebook
notebook:
	$(PYTHON) tools/percent_to_ipynb.py lab_transformer.py lab_transformer.ipynb
check:
	$(PYTHON) tools/check_lab.py
check-pretrained:
	$(PYTHON) tools/check_lab.py --pretrained
