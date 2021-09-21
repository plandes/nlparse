# makefile for the Demonstrates resource libraries. app

MTARG=		target
SENT=		'He was George Washington, the first president of the United States.'

.PHONY:		run
run:
		@echo "---------- lemma.conf"
		./run.py parse $(SENT) -c lemma.conf | head -12
		@echo "--<all tokens provided; trimmed for brevity>--"
		@echo "---------- terse.conf (filter punctuation, stop words)"
		./run.py parse $(SENT) -c terse.conf
		@echo "---------- terse.conf (filter punctuation, stop words)"
		./run.py dataframe $(SENT) -c terse.conf

.PHONY:		clean
clean:
		rm -fr $(MTARG)
		find . -type d -name __pycache__ -prune -exec rm -r {} \;
