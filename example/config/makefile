# makefile for the Demonstrates resource libraries. app

MTARG=		target
PROG=		./run.py
SENT=		'He was George Washington, the first president of the United States.'

.PHONY:		run
run:
		@echo "---------- no configuration give, use defaults"
		$(PROG) parse $(SENT) | head -12
		@echo "--<all tokens provided; trimmed for brevity>--"
		@echo "---------- lemma.conf (lemmatize each token)"
		$(PROG) parse $(SENT) -c lemma.conf | head -12
		@echo "--<all tokens provided; trimmed for brevity>--"
		@echo "---------- terse.conf (filter punctuation, stop words)"
		$(PROG) parse $(SENT) -c terse.conf
		@echo "---------- terse.conf (filter punctuation, stop words)"
		$(PROG) csv $(SENT) -c terse.conf

.PHONY:		help
help:
		$(PROG) -h

.PHONY:		clean
clean:
		rm -fr $(MTARG)
		find . -type d -name __pycache__ -prune -exec rm -r {} \;
