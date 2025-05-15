import logging
from io import BytesIO
import pickle
from zensols.nlp import FeatureSentence
from util import TestBase

logger = logging.getLogger(__name__)


class TestFeatureDocParse(TestBase):
    SENT_TEXT_SPACE = """I am a 	citizen of 

the United States of America."""

    def test_basic_parse(self):
        dp = self.fac.instance('doc_parser')
        doc = dp.parse(self.sent_text)
        self.assertEqual(self.def_parse, tuple(doc.norm_token_iter()))

    def test_whitespace_default_parse(self):
        parser = self.fac('doc_parser_default')
        fdoc = parser.parse(self.SENT_TEXT_SPACE)
        should = ('I', 'am', 'a', '\t', 'citizen', 'of', '\n\n',
                  'the United States of America', '.')
        self.assertEqual(should, tuple(fdoc.norm_token_iter()))

    def test_whitespace_parse(self):
        parser = self.fac('doc_parser')
        fdoc = parser.parse(self.SENT_TEXT_SPACE)
        self.assertEqual(self.def_parse, tuple(fdoc.norm_token_iter()))

    def _test_token_iter(self, doc):
        should_s1 = ('I', 'am', 'a', 'citizen', 'of','the United States of America', '.')
        should_s2 = ('My', 'name', 'is', 'Paul Landes', '.')
        should = should_s1 + should_s2

        self.assertEqual(2, len(doc))
        self.assertEqual(len(should), doc.token_len)
        self.assertEqual(len(should_s1), doc.sents[0].token_len)
        self.assertEqual(len(should_s2), doc.sents[1].token_len)
        self.assertEqual(len(should_s2), len(doc.sents[1]))

        self.assertEqual(should_s1, tuple(doc.sents[0].norm_token_iter()))
        self.assertEqual(should_s1, tuple(doc.sents[0].norm_token_iter()))
        self.assertEqual(should_s2, tuple(doc.sents[1].norm_token_iter()))
        self.assertEqual(should_s1, tuple(map(lambda t: t.norm, doc.sents[0].tokens)))

        for i in range(2):
            self.assertEqual(i, len(tuple(doc.sent_iter(i))))

        self.assertEqual(' '.join(should_s1[:-1]) + '.', next(doc.sent_iter()).text)

        sent = doc.to_sentence()
        self.assertEqual(FeatureSentence, type(sent))
        self.assertEqual(should, tuple(sent.norm_token_iter()))

        sent = doc.to_sentence(limit=1)
        self.assertEqual(should_s1, tuple(sent.norm_token_iter()))

        sent = doc.to_sentence(limit=9999)
        self.assertEqual(should, tuple(sent.norm_token_iter()))

        # sent = doc.to_sentence(1, 2)
        # self.assertEqual(should_s2, tuple(sent.norm_token_iter()))

        for i in range(len(should_s1)):
            self.assertEqual(should_s1[:i], tuple(doc.sents[0].norm_token_iter(i)))
        for i in range(len(should_s2)):
            self.assertEqual(should_s2[:i], tuple(doc.sents[1].norm_token_iter(i)))

        self.assertEqual(should, tuple(doc.norm_token_iter()))
        self.assertEqual(should, tuple(doc.norm_token_iter()))
        self.assertEqual(should, tuple(map(lambda t: t.norm, doc.tokens)))
        for i in range(len(should)):
            self.assertEqual(should[:i], tuple(doc.norm_token_iter(i)))

        should = ('PRP', 'VBP', 'DT', 'NN', 'IN', 'DT', '.', 'PRP$', 'NN',
                  'VBZ', 'NNP', '.')
        self.assertEqual(should, tuple(map(lambda t: t.tag_, doc.tokens)))

    def test_sentence_reindex(self):
        #       0    1   2   3   4   6         8   9 (no pronouns)
        #       0    1   2   3     5 6      7  8     (no punctuation)
        #       0    1   2   3   4 5 6      7  8   9 (all tokens)
        text = 'John hit the ball. He moved it fast.'
        mapper = self.fac('sent_idx_token_mapper')
        dp = self.fac('sent_idx_doc_parser')

        mapper.remove_pronouns = False
        mapper.remove_punctuation = False
        doc = dp(text)
        doc_should = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4)
        self.assertEqual(doc_should,
                         tuple(map(lambda t: t.i_sent, doc.token_iter())))
        sent = doc.to_sentence()
        self.assertEqual(doc_should,
                         tuple(map(lambda t: t.i_sent, sent.token_iter())))
        sent = doc.to_sentence(contiguous_i_sent=True)
        self.assertEqual((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                         tuple(map(lambda t: t.i_sent, sent.token_iter())))

        mapper.remove_pronouns = True
        mapper.remove_punctuation = False
        doc = dp(text)
        doc_should = (0, 1, 2, 3, 4, 1, 3, 4)
        self.assertEqual(doc_should,
                         tuple(map(lambda t: t.i_sent, doc.token_iter())))
        sent = doc.to_sentence()
        self.assertEqual(doc_should,
                         tuple(map(lambda t: t.i_sent, sent.token_iter())))

        sent = doc.to_sentence(contiguous_i_sent=True)
        self.assertEqual((0, 1, 2, 3, 4, 6, 8, 9),
                         tuple(map(lambda t: t.i_sent, sent.token_iter())))

        mapper.remove_pronouns = False
        mapper.remove_punctuation = True
        doc = dp(text)
        sent = doc.to_sentence(contiguous_i_sent=True)
        self.assertEqual((0, 1, 2, 3, 5, 6, 7, 8),
                         tuple(map(lambda t: t.i_sent, sent.token_iter())))

    def test_token_iteration(self):
        parser = self.fac('doc_parser')
        doc = parser.parse(self.sent_text2)
        self._test_token_iter(doc)

    def test_token_iteration_pickle(self):
        parser = self.fac('doc_parser')
        doc = parser.parse(self.sent_text2)
        b = BytesIO()
        pickle.dump(doc, b)
        b.seek(0)
        doc_new = pickle.load(b)
        self._test_token_iter(doc_new)

    def _test_sent_parsing(self, text: str, parser_name: str, n_sents: int,
                           has_lead_space: bool, has_trail_space: bool = False):
        parser = self.fac(parser_name)
        sents = parser.parse(text)
        self.assertEqual(n_sents, len(sents))
        n_sent = 0
        should = "I'm Paul Landes and I live in the United States."
        if has_trail_space:
            should = should + '  '
        self.assertEqual(should, sents[n_sent].text)
        self.assertEqual(("I", "'m", 'Paul Landes', 'and',
                          'I', 'live', 'in', 'the United States', '.'),
                         tuple(sents[n_sent].norm_token_iter()))
        n_sent += 1
        if n_sents == 3:
            self.assertEqual((), tuple(sents[n_sent].norm_token_iter()))
            n_sent += 1
        should = "I'm done."
        if has_lead_space:
            should = ' ' + should
        self.assertEqual(should, sents[n_sent].text)
        self.assertEqual(('I', "'m", 'done', '.'),
                         tuple(sents[n_sent].norm_token_iter()))

    def test_sent_parsing(self):
        text = "I'm Paul Landes and I live in the United States. I'm done."
        self._test_sent_parsing(text, 'doc_parser', 2, False)
        text = "I'm Paul Landes and I live in the United States.  I'm done."
        self._test_sent_parsing(text, 'strip_sent_doc_parser', 2, False)
        self._test_sent_parsing(text, 'doc_parser_no_remove_sents', 2, False, True)

    def test_feature_subset(self):
        parser = self.fac('doc_parser_default')
        fdoc = parser.parse(self.SENT_TEXT_SPACE)

        for tok in fdoc.tokens:
            self.assertEqual(28, len(tok.asdict()))

        parser = self.fac('doc_parser_feat_subset')
        fdoc = parser.parse(self.SENT_TEXT_SPACE)
        for tok in fdoc.tokens:
            self.assertEqual(7, len(tok.asdict()))

        parser = self.fac('doc_parser_feat_no_exist')
        with self.assertRaises(AttributeError):
            fdoc = parser.parse(self.SENT_TEXT_SPACE)

    def test_entity(self):
        parser = self.fac('doc_parser_split_ents')
        doc = parser.parse(self.sent_text2)
        ents = doc.entities
        self.assertEqual(len(ents), 2)
        should = '(<the United States of America>, <Paul Landes>)'
        self.assertEqual(should, str(ents))

    def test_entity_pickled(self):
        parser = self.fac('doc_parser_split_ents')
        doc = parser.parse(self.sent_text2)
        bio = BytesIO()
        pickle.dump(doc, bio)
        doc = None
        bio.seek(0)
        doc2 = pickle.load(bio)
        ents = doc2.entities
        self.assertEqual(len(ents), 2)
        should = '(<the United States of America>, <Paul Landes>)'
        self.assertEqual(should, str(ents))
