"""Demonstrates resource libraries by using the Zensols natural language
processing library to parse English text.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import sys
from pathlib import Path
from spacy.tokens.doc import Doc
from zensols.config import Configurable
from zensols.nlp import FeatureDocument, FeatureDocumentParser
import logging

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Parses natural language text.

    """
    CLI_META = {'option_excludes': {'config', 'doc_parser'},
                'option_overrides': {'output_file': {'long_name': 'out'}}}

    config: Configurable
    doc_parser: FeatureDocumentParser

    def show_config(self):
        """Print out the configuration."""
        self.config.write()

    def csv(self, sentence: str, output_file: Path = None):
        """Create and print a Pandas (if installed) dataframe of feature.

        :param sentence: the sentene to parse

        :output_file: the CSV file to create, otherwise print to standard out

        """
        import pandas as pd
        from zensols.nlp.dataframe import FeatureDataFrameFactory

        fac = FeatureDataFrameFactory()
        doc = self.doc_parser.parse(sentence)
        df: pd.DataFrame = fac(doc)
        if output_file is None:
            try:
                from tabulate import tabulate
                print(tabulate(df))
            except Exception as e:
                logger.error(f'tabulate not installed: {e}--using CSV')
                df.to_csv(sys.stdout, index=False)
        else:
            df.to_csv(output_file, index=False)

    def parse(self, sentence: str, token_length: int = -1):
        """Parse a sentence.

        :param sentence: the sentene to parse


        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing: {sentence}')
        doc: FeatureDocument = self.doc_parser(sentence)
        #print(doc)
        token_length = sys.maxsize if token_length == -1 else token_length
        doc.write(n_tokens=token_length)

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
        doc: Doc = self.doc_parser.parse_spacy_doc(sentence)
        print(doc)
        print('-' * 10, 'token POS, stop words:')
        for tok in doc:
            print(tok, tok.tag_, tok.is_stop)
        print('-' * 10, 'token features:')
        doc: FeatureDocument = self.doc_parser(sentence)
        toks = doc.tokens
        print(toks)
        for feat in toks:
            print(f'{feat} {type(feat)}')
            feat.write_attributes(depth=1, feature_ids=(*feat.WRITABLE_FEATURE_IDS, 'sent_i'))
            print('-' * 5)
        print(', '.join(doc.norm_token_iter()))
        print('-' * 10)
