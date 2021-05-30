from dataclasses import dataclass
from zensols.nlp import LanguageResource, FeatureDocumentParser


@dataclass
class Application(object):
    """Demonstrate the NLP parsing API.

    """
    CONFIG_CLI = {'option_includes': {}}

    langres: LanguageResource
    lc_langres: LanguageResource
    doc_parser: FeatureDocumentParser

    def run(self):
        sent = 'California is part of the United States.  Next sentence.'

        doc = self.langres.parse(sent)
        print(type(doc))
        for tok in doc:
            print(tok, tok.tag_, tok.is_stop)
        print('-' * 10)
        feats = self.langres.features(doc)
        for feat in feats:
            print(f'{feat} {type(feat)}')
            feat.write(depth=1)
            print('-' * 5)
            det = feat.detach()
            print(f'detached: {type(det)}: {det.asdict()}')
            print('-' * 5)
        print(', '.join(self.langres.normalized_tokens(doc)))
        print('-' * 10)

        doc = self.lc_langres.parse(sent)
        print(', '.join(self.lc_langres.normalized_tokens(doc)))
        print('-' * 10)

        doc = self.doc_parser.parse(sent)
        doc.write()
