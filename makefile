## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE =		python
PROJ_MODULES =		git python-doc python-doc-deploy
PY_DEP_POST_DEPS +=	nlpdeps
SPACY_MODELS +=		sm

include ./zenbuild/main.mk

# https://spacy.io/models/en
.PHONY:			nlpdeps
nlpdeps:
			for i in $(SPACY_MODELS) ; do \
				$(PYTHON_BIN) -m spacy download en_core_web_$${i} ; \
			done
