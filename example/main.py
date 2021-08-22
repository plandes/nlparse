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
    cli = ApplicationFactory('nlparse', StringIO(CONFIG),
                             reload_pattern=r'^app')
    cli.invoke()


if (__name__ == '__main__'):
    # when running from a shell, run the CLI entry point
    import __main__ as mmod
    if hasattr(mmod, '__file__'):
        main()
    # otherwise, assume a Python REPL and run the prototyping method
    else:
        print('--> proto')
        main()
