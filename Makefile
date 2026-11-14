PYTHON ?= python3

.PHONY: all notebook figures check check-pretrained check-switch-pretrained
all: figures notebook
notebook:
	$(PYTHON) tools/percent_to_ipynb.py lab_transformer.py lab_transformer.ipynb
check:
	$(PYTHON) tools/check_lab.py
check-pretrained:
	$(PYTHON) tools/check_lab.py --pretrained

figures:
	$(PYTHON) tools/plot_figures.py

check-switch-pretrained:
	$(PYTHON) tools/check_lab.py --switch-pretrained
