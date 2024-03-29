#!/usr/bin/env python

from zensols.cli import CliHarness

if (__name__ == '__main__'):
    sent = 'Barak Obama was the 44th president of the United States.'
    CliHarness(
        app_config_resource='app.conf',
        proto_args=['parse', '-c', 'terse.conf', sent],
        proto_factory_kwargs={'reload_pattern': '^app'},
    ).run()
