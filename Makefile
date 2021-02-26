SHELL=bash
python=python
pip=pip
tests=./test

install: 
	- $(python) setup.py build
	- $(python) setup.py build_ext --inplace
	- $(python) -m pip install --user .

check_build_reqs:
	@$(python) -c 'import pytest' \
                || ( printf "$(redpip)Build requirements are missing. Run 'make prepare' to install them.$(normal)" ; false )

test: check_build_reqs
	$(python) -m pytest -vv $(tests)
