from util import TestBase

if 0:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('zensols.nlp.combine').setLevel(logging.DEBUG)


class TestDocCombine(TestBase):
    def test_align(self):
        doc_parser = self.fac.instance('doc_parser_combiner')
        doc = doc_parser(self.sent_text2)
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

    def test_align_reverse(self):
        doc_parser = self.fac.instance('doc_parser_combiner_reverse')
        doc = doc_parser(self.sent_text2)
        should = \
            ((0, 'I', '<none>'), (2, 'am', '<none>'), (5, 'a', '<none>'),
             (7, 'citizen', '<none>'), (15, 'of', '<none>'),
             (18, 'the United States of America', 'GPE'), (46, '.', '<none>'),
             # sent 2
             (48, 'My', '<none>'), (51, 'name', '<none>'), (56, 'is', '<none>'),
             (59, 'Paul Landes', 'PERSON'), (70, '.', '<none>'))
        toks = tuple(map(lambda t: (t.idx, t.norm, t.ent_), doc.token_iter()))
        if 0:
            print()
            print(toks)
        self.assertEqual(should, toks)
