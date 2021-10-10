#!/usr/bin/env python

from io import StringIO
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocument

CONFIG = """
[import]
sections = list: imp_conf

# import the ``zensols.nlp`` library
[imp_conf]
type = importini
config_files = list: resource(zensols.nlp): resources/obj.conf

# override the parse to keep only the norm, ent
[doc_parser]
token_feature_ids = eval: set('ent_ tag_'.split())
"""

if __name__ == '__main__':
    fac = ImportConfigFactory(ImportIniConfig(StringIO(CONFIG)))
    doc_parser = fac('doc_parser')
    sent = 'He was George Washington and first president of the United States.'
    doc: FeatureDocument = doc_parser(sent)
    for tok in doc.tokens:
        tok.write()
