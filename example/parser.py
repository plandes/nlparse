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
"""


def main():
    print()
    cli = ApplicationFactory('nlparse', StringIO(CONFIG))
    cli.invoke()


if __name__ == '__main__':
    main()
