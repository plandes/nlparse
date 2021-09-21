"""Demonstrates resource libraries by using the Zensols natural language
processing library to parse English text.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import sys
from zensols.nlp import FeatureDocument, FeatureDocumentParser
import logging

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Parses natural language text.

    """
    CONFIG_CLI = {'option_includes': {}}

    doc_parser: FeatureDocumentParser

    def dataframe(self, sent: str):
        import pandas as pd
        from zensols.nlp.dataframe import FeatureDataFrameFactory
        fac = FeatureDataFrameFactory()
        doc = self.doc_parser.parse(sent)
        df: pd.DataFrame = fac(doc)
        try:
            from tabulate import tabulate
            print(tabulate(df))
        except Exception as e:
            logger.error(f'tabulate not installed: {e}--using CSV')
            df.to_csv(sys.stdout, index=False)

    def parse(self, sentence: str):
        """Parse a sentence.

        :param sentence: the sentene to parse

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing: {sentence}')
        doc: FeatureDocument = self.doc_parser(sentence)
        print(doc)
        doc.write(n_tokens=sys.maxsize)

    def json(self, sentence: str):
        """Parse a sentence and output JSON.

        :param sentence: the sentene to parse

        """
        doc: FeatureDocument = self.doc_parser(sentence)
        for feat in doc:
            print(feat.asjson(indent=4))

    def detailed(self, sentence: str):
        """Deeper API to demonstrate language resources.

        :param sentence: the sentene to parse

        """
        doc = self.doc_parser.langres.parse(sentence)
        print(f'document ({type(doc)}:')
        print(doc)
        print('-' * 10, 'token POS, stop words:')
        for tok in doc:
            print(tok, tok.tag_, tok.is_stop)
        print('-' * 10, 'token features:')
        feats = self.langres.features(doc)
        print(tuple(feats))
        for feat in feats:
            print(f'{feat} {type(feat)}')
            feat.write(depth=1, field_ids=(*feat.WRITABLE_FIELD_IDS, 'sent_i'))
            print('-' * 5)
            # if print_dict:
            #     print(feat.asdict())
            #     print('-' * 5)
        print(', '.join(self.langres.normalized_tokens(doc)))
        print('-' * 10)

        doc = self.lc_langres.parse(sentence)
        print(', '.join(self.lc_langres.normalized_tokens(doc)))
        print('-' * 10)

        doc = self.doc_parser.parse(sentence)
        doc.write()
