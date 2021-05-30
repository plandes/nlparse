#!/usr/bin/env python

from zensols.cli import ApplicationFactory
from io import StringIO


CONFIG = """
[cli]
class_name = zensols.cli.ActionCliManager
apps = list: app

[import]
files = parser.conf

[app]
class_name = app.Application
langres = instance: langres
lc_langres = instance: lc_langres
doc_parser = instance: doc_parser
"""


def main():
    print()
    if 0:
        import logging
        logging.basicConfig(level=logging.WARN)
        logging.getLogger('zensols.nlp').setLevel(logging.DEBUG)
    cli = ApplicationFactory('nlparse', StringIO(CONFIG), reload_factory=True)
    cli.invoke()


if __name__ == '__main__':
    main()


main()
