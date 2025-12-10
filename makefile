## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy
PY_DOC_DIST_NAME =	nlparse


PY_TEST_GLOB =		test_doc_compose.py


## Includes
#
include ./zenbuild/main.mk
