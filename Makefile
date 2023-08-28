SHELL=bash
python=python
pip=pip
tests=./test
version:=$(shell $(python) version.py)
sdist_name:=cassiopeia-$(version).tar.gz

develop:
	$(pip) install -e .

clean_develop:
	- $(pip) uninstall -y cassiopeia
	- rm -rf *.egg-info

clean_sdist:
	- rm -rf dist

clean: clean_develop clean_pypi

install:
	- $(python) -m pip install .

check_build_reqs:
	@$(python) -c 'import pytest' \
                || ( printf "$(redpip)Build requirements are missing. Run 'make prepare' to install them.$(normal)" ; false )

test: check_build_reqs
	$(python) -m pytest -vv $(tests)