#!/usr/bin/env python

from io import StringIO
from zensols.cli import CliHarness

CONFIG = """
[cli]
class_name = zensols.cli.ActionCliManager
apps = list: log_cli, list_cli, config_cli, app

[log_cli]
class_name = zensols.cli.LogConfigurator
log_name = app
level = debug

[list_cli]
class_name = zensols.cli.ListActions

[config_cli]
class_name = zensols.cli.ConfigurationImporter
# allow overriding the ``zensols.nlp`` resource library
override = True
expect = False
#default = path: default.conf

[import]
references = default
sections = imp_conf

# import the ``zensols.nlp`` library
[imp_conf]
type = importini
config_files = list: resource(zensols.nlp): resources/obj.conf

[app]
class_name = app.Application

# after we import the ``zensols.nlp`` library (above), we have access to it's
# configured FeatureDocumentParser with overriden configuration in
# ``terse.conf``
doc_parser = instance: doc_parser
"""


CliHarness(
    app_config_resource=StringIO(CONFIG),
    proto_args=['-c', 'terse.conf',
                'Barak Obama was the 44th president of the United States.'],
    proto_factory_kwargs={'reload_pattern': '^app'},
).run()
