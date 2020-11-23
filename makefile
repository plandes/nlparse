## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python
PROJ_MODULES=	git python-doc python-doc-deploy

SPACY_MODELS +=	sm

include ./zenbuild/main.mk

# https://spacy.io/models/en
.PHONY:		nlpdeps
nlpdeps:	pydeps
		for i in $(SPACY_MODELS) ; do \
			$(PYTHON_BIN) -m spacy download en_core_web_$${i} ; \
		done

.PHONY:		testparse
testparse:
		make PY_SRC_TEST_PKGS=test_nlparse.TestParse test

.PHONY:		testthird
testthird:
		make PY_SRC_TEST_PKGS=test_thirdparty.TestThirdParty test
