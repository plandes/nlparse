from util import TestBase
from spacy.tokens.doc import Doc
from zensols.nlp import (
    FeatureDocument, FeatureDocumentParser, ParseError
)


class TestDocCombine(TestBase):
    def test_ent_splitter(self):
        doc_parser: FeatureDocumentParser = self.fac.instance(
            'doc_parser_split_ents')
        doc: Doc = doc_parser(self.sent_text2)
        toks = tuple(doc.norm_token_iter())
        should_ents = \
            (('I', '-<N>-'), ('am', '-<N>-'), ('a', '-<N>-'),
             ('citizen', '-<N>-'), ('of', '-<N>-'), ('the', 'GPE'),
             ('United', 'GPE'), ('States', 'GPE'), ('of', 'GPE'),
             ('America', 'GPE'), ('.', '-<N>-'),
             # sent 2
             ('My', '-<N>-'), ('name', '-<N>-'),
             ('is', '-<N>-'), ('Paul', 'PERSON'), ('Landes', 'PERSON'),
             ('.', '-<N>-'))
        should = tuple(map(lambda x: x[0], should_ents))
        self.assertEqual(should, tuple(toks))
        tents = tuple(map(lambda t: (t.norm, t.ent_),
                          doc_parser(self.sent_text2).token_iter()))
        if 0:
            print(tents)
        self.assertEqual(should_ents, tents)

    def test_align(self):
        doc_parser: FeatureDocumentParser = self.fac.instance('doc_parser_combiner')
        doc: FeatureDocument = doc_parser(self.sent_text2)
        should = \
            ((0, 'I', '-<N>-'), (2, 'am', '-<N>-'), (5, 'a', '-<N>-'),
             (7, 'citizen', '-<N>-'), (15, 'of', '-<N>-'), (18, 'the', 'GPE'),
             (22, 'United', 'GPE'), (29, 'States', 'GPE'), (36, 'of', 'GPE'),
             (39, 'America', 'GPE'), (46, '.', '-<N>-'),
             # sent 2
             (48, 'My', '-<N>-'), (51, 'name', '-<N>-'),
             (56, 'is', '-<N>-'), (59, 'Paul', 'PERSON'),
             (64, 'Landes', 'PERSON'), (70, '.', '-<N>-'))
        toks = tuple(map(lambda t: (t.idx, t.norm, t.ent_), doc.token_iter()))
        if 0:
            print()
            print(toks)
        self.assertEqual(should, toks)

        ents = doc.entities
        should = '(<the United States of America>, <Paul Landes>)'
        if 0:
            print()
            print(ents)
        self.assertEqual(should, str(ents))

    def test_align_reverse(self):
        doc_parser: FeatureDocumentParser = self.fac.instance('doc_parser_combiner_reverse')
        doc: FeatureDocument = doc_parser(self.sent_text2)
        should = \
            ((0, 'I', '-<N>-'), (2, 'am', '-<N>-'), (5, 'a', '-<N>-'),
             (7, 'citizen', '-<N>-'), (15, 'of', '-<N>-'),
             (18, 'the', 'GPE'), (22, 'United', 'GPE'),
             (29, 'States', 'GPE'), (36, 'of', 'GPE'), (39, 'America', 'GPE'),
             (46, '.', '-<N>-'),
             # sent 2
             (48, 'My', '-<N>-'), (51, 'name', '-<N>-'), (56, 'is', '-<N>-'),
             (59, 'Paul', 'PERSON'), (64, 'Landes', 'PERSON'), (70, '.', '-<N>-'))
        toks = tuple(map(lambda t: (t.idx, t.norm, t.ent_), doc.token_iter()))
        if 0:
            print()
            print(toks)
        self.assertEqual(should, toks)

    def test_align_reverse2(self):
        doc_parser: FeatureDocumentParser = self.fac.instance('doc_parser_combiner_reverse_2')
        # this doesn't work because one (primary multi-token span) to many
        # (replica splits entities) mapping isn't supported
        with self.assertRaisesRegex(ParseError, r'^Mismatch tokens: the United States of America\(norm=the United States of America\) != the\(norm=the\)'):
            doc_parser(self.sent_text2)
