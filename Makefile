
install:
	python3 -m pip install pylint
	python3 -m pip install flake8
	python3 -m pip install black

pylint:
	pylint *.py --disable=too-few-public-methods

flake8:
	flake8 .

format_black:
	black *.py