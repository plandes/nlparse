## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-doc python-doc-deploy
PY_DEP_POST_DEPS +=	modeldeps
SPACY_MODELS +=		sm md lg
PIP_ARGS +=		--use-deprecated=legacy-resolver

#PY_SRC_TEST_PAT ?=	'test_doc_c*.py'

include ./zenbuild/main.mk

# https://spacy.io/models/en
.PHONY:			allmodels
allmodels:
			@for i in $(SPACY_MODELS) ; do \
				echo "installing $$i" ; \
				$(PYTHON_BIN) -m spacy download en_core_web_$${i} ; \
			done

.PHONY:			modeldeps
modeldeps:
			$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-model.txt

.PHONY:			uninstalldeps
uninstalldeps:
			$(PYTHON_BIN) -m pip freeze | grep spacy | sed 's/\([^= ]*\).*/\1/' | xargs pip uninstall -y
