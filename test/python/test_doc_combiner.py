from util import TestBase
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from zensols.nlp import (
    LanguageResource, FeatureDocument, FeatureDocumentParser, ParseError
)


if 0:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('zensols.nlp.combine').setLevel(logging.DEBUG)


class TestDocCombine(TestBase):
    def test_ent_splitter(self):
        doc_parser: FeatureDocumentParser = self.fac.instance(
            'doc_parser_split_ents')
        lr = doc_parser.langres
        doc: Doc = lr(self.sent_text2)
        toks = tuple(lr.normalized_tokens(doc))
        should_ents = \
            (('I', '<none>'), ('am', '<none>'), ('a', '<none>'),
             ('citizen', '<none>'), ('of', '<none>'), ('the', 'GPE'),
             ('United', 'GPE'), ('States', 'GPE'), ('of', 'GPE'),
             ('America', 'GPE'), ('.', '<none>'),
             # sent 2
             ('My', '<none>'), ('name', '<none>'),
             ('is', '<none>'), ('Paul', 'PERSON'), ('Landes', 'PERSON'),
             ('.', '<none>'))
        should = tuple(map(lambda x: x[0], should_ents))
        self.assertEqual(should, tuple(toks))
        tents = tuple(map(lambda t: (t.norm, t.ent_),
                          doc_parser(self.sent_text2).token_iter()))
        if 0:
            print(tents)
        self.assertEqual(should_ents, tents)

    def test_align(self):
        doc_parser = self.fac.instance('doc_parser_combiner')
        doc: FeatureDocument = doc_parser(self.sent_text2)
        should = \
            ((0, 'I', '<none>'), (2, 'am', '<none>'), (5, 'a', '<none>'),
             (7, 'citizen', '<none>'), (15, 'of', '<none>'), (18, 'the', 'GPE'),
             (22, 'United', 'GPE'), (29, 'States', 'GPE'), (36, 'of', 'GPE'),
             (39, 'America', 'GPE'), (46, '.', '<none>'),
             # sent 2
             (48, 'My', '<none>'), (51, 'name', '<none>'),
             (56, 'is', '<none>'), (59, 'Paul', 'PERSON'),
             (64, 'Landes', 'PERSON'), (70, '.', '<none>'))
        toks = tuple(map(lambda t: (t.idx, t.norm, t.ent_), doc.token_iter()))
        if 0:
            print()
            print(toks)
        self.assertEqual(should, toks)

        ents = doc.entities
        should = '((<the>, <United>, <States>, <of>, <America>), (<Paul>, <Landes>))'
        if 0:
            print()
            print(ents)
        self.assertEqual(should, str(ents))

    def test_align_reverse(self):
        doc_parser = self.fac.instance('doc_parser_combiner_reverse')
        doc = doc_parser(self.sent_text2)
        should = \
            ((0, 'I', '<none>'), (2, 'am', '<none>'), (5, 'a', '<none>'),
             (7, 'citizen', '<none>'), (15, 'of', '<none>'),
             (18, 'the', 'GPE'), (22, 'United', 'GPE'),
             (29, 'States', 'GPE'), (36, 'of', 'GPE'), (39, 'America', 'GPE'),
             (46, '.', '<none>'),
             # sent 2
             (48, 'My', '<none>'), (51, 'name', '<none>'), (56, 'is', '<none>'),
             (59, 'Paul', 'PERSON'), (64, 'Landes', 'PERSON'), (70, '.', '<none>'))
        toks = tuple(map(lambda t: (t.idx, t.norm, t.ent_), doc.token_iter()))
        if 0:
            print()
            print(toks)
        self.assertEqual(should, toks)

    def test_align_reverse2(self):
        doc_parser = self.fac.instance('doc_parser_combiner_reverse_2')
        # this doesn't work because one (primary multi-token span) to many
        # (replica splits entities) mapping isn't supported
        with self.assertRaisesRegex(ParseError, r'^Mismatch tokens: the United States of America\(norm=the United States of America\) != the\(norm=the\)'):
            doc_parser(self.sent_text2)

    # def test_first(self):
    #     print()
    #     doc_parser = self.fac.instance('doc_parser_combiner')
    #     doc = doc_parser('The United States is where I live.')
    #     doc.sents[0].write()
