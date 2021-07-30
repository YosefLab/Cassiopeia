SHELL=bash
python=python
pip=pip
tests=./test
version:=$(shell $(python) version.py)
sdist_name:=PlotMAPQ-$(version).tar.gz

install: 
	- $(python) setup.py build
	- $(python) setup.py build_ext --inplace
	- $(python) -m pip install --user .

check_build_reqs:
	@$(python) -c 'import pytest' \
                || ( printf "$(redpip)Build requirements are missing. Run 'make prepare' to install them.$(normal)" ; false )

test: check_build_reqs
	$(python) -m pytest -vv $(tests)

pypi: clean clean_sdist
	set -x \
	&& $(python) setup.py sdist bdist_wheel \
	&& twine check dist/* \
	&& twine upload --repository-url https://test.pypi.org/legacy/ dist/*

clean_pypi:
	- rm -rf build/