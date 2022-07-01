import logging
import unittest
import json
from spacy.tokens import Doc
from zensols.util.log import loglevel
from zensols.config import ImportConfigFactory
from zensols.nlp import FeatureDocumentParser
from config import AppConfig

logger = logging.getLogger(__name__)


def rec_sort(x):
    if isinstance(x, list) or isinstance(x, tuple):
        x = sorted(list(map(rec_sort, x)))
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = rec_sort(v)
        x = sorted(x.items())
    return x


class TestParse(unittest.TestCase):
    def setUp(self):
        self.maxDiff = 999999
        self.config = AppConfig()
        self.fac = ImportConfigFactory(self.config, shared=False)
        self.doc_parser = self.fac('default_doc_parser')
        self.sent = 'Dan throws the ball.'

    def test_parse(self):
        doc_parser = self.doc_parser
        self.assertTrue(isinstance(doc_parser, FeatureDocumentParser))
        doc: Doc = doc_parser.parse_spacy_doc(self.sent)
        dd = self.doc_parser.get_dictable(doc)
        res = dd.asdict()
        with open(self.config.parse_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))

    def test_feature(self):
        tnfac = ImportConfigFactory(self.config, shared=False)
        tn = tnfac.instance('default_token_normalizer')
        doc_parser = self.fac('default_doc_parser', token_normalizer=tn)
        self.assertEqual('MapTokenNormalizer: embed=True, lemma_token_mapper', str(tn))
        fd = doc_parser(self.sent)
        res = fd.asdict()
        if 0:
            with open(self.config.feature_path, 'w') as f:
                f.write(fd.asjson(indent=4))
        with open(self.config.feature_path) as f:
            c = json.load(f)
        self.assertEqual(rec_sort(c), rec_sort(res))
        tn = tnfac.instance('nonorm_token_normalizer')
        doc_parser = self.fac('default_doc_parser', token_normalizer=tn)
        res = tuple(map(lambda x: x.norm, doc_parser(self.sent).token_iter()))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def _from_token_norm(self, sent, norm_name):
        doc_parser = self.fac('default_doc_parser')
        doc_parser.token_normalizer = self.fac(norm_name)
        doc = doc_parser(sent)
        return doc.token_iter()

    def test_map(self):
        sent = 'I am a citizen of the United States of America.'
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(map(lambda t: t.norm, self._from_token_norm(
                             sent, 'map_filter_token_normalizer'))))
        self.assertEqual(('am', 'citizen', 'of', 'the United States of America'),
                         tuple(map(lambda t: t.norm, self._from_token_norm(
                             sent, 'map_filter_pron_token_normalizer'))))
        self.assertEqual(('i', 'am', 'a', 'citizen', 'of', 'the united states of america', '.'),
                         tuple(map(lambda t: t.norm, self._from_token_norm(
                             sent, 'map_lower_token_normalizer'))))
        self.assertEqual(('citizen', 'United', 'States', 'America'),
                         tuple(map(lambda t: t.norm, self._from_token_norm(
                             sent, 'map_embed_token_normalizer'))))
        self.assertEqual(('citizen', 'the_united_states_of_america'),
                         tuple(map(lambda t: t.norm, self._from_token_norm(
                             sent, 'map_filter_subs_token_normalizer'))))

    def test_disable(self):
        import warnings
        # since spacy 3.1, warnings are thrown when certain components are
        # disabled in the pipline
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r"^\[W108\] The rule-based lemmatizer did not find POS annotation for the token.*Check that your pipeline")
            #lr = self.lr
            dis_dp = self.fac('disable_tagger_doc_parser')
            doc = self.doc_parser.parse_spacy_doc(self.sent)
            tags = tuple(map(lambda t: t.tag_, doc))
            self.assertEqual(('NNP', 'VBZ', 'DT', 'NN', '.'), tags)
            self.assertEqual('tagger parser'.split(), dis_dp.disable_component_names)
            # spacy warns about trying to add POS tags
            with loglevel('spacy', logging.ERROR):
                doc_dis = dis_dp.parse_spacy_doc(self.sent)
            no_tags = tuple(map(lambda t: t.tag_, doc_dis))
            self.assertEqual(('', '', '', '', ''), no_tags)

    def test_filter_features(self):
        tnfac = ImportConfigFactory(self.config)
        dp = self.fac('default_doc_parser', token_normalizer=tnfac.instance('feature_no_filter_token_normalizer'))
        feats = dp('I am a citizen of the United States of America.').token_iter()
        self.assertEqual(('I', 'am', 'a', 'citizen', 'of', 'the United States of America', '.'),
                         tuple(map(lambda f: f.norm, feats)))
        dp = self.fac('default_doc_parser', token_normalizer=tnfac.instance('feature_default_filter_token_normalizer'))
        feats = dp.parse('I am a citizen of the United States of America.').token_iter()
        self.assertEqual(('I', 'am', 'citizen', 'of', 'the United States of America'),
                         tuple(map(lambda f: f.norm, feats)))
        dp = self.fac('default_doc_parser', token_normalizer=tnfac.instance('feature_stop_filter_token_normalizer'))
        feats = dp.parse('I am a citizen of the United States of America.').token_iter()
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(map(lambda f: f.norm, feats)))

    def test_space(self):
        sent = '''Dan throws
the ball.'''
        tn = self.fac('nonorm_token_normalizer')
        dp = self.fac('default_doc_parser', token_normalizer=tn)
        doc = dp.parse(sent)
        res = tuple(map(lambda x: x.norm, doc.token_iter()))
        self.assertEqual(('Dan', 'throws', '\n', 'the', 'ball', '.'), res)

        tn = self.fac('map_filter_space_token_normalizer')
        dp = self.fac('default_doc_parser', token_normalizer=tn)
        doc = dp.parse(sent)
        res = tuple(map(lambda x: x.norm, doc.token_iter()))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def test_tok_boundaries(self):
        tn = self.fac('nonorm_token_normalizer')
        dp = self.fac('default_doc_parser', token_normalizer=tn)
        doc = dp.parse('id:1234')
        res = tuple(map(lambda x: x.norm, doc.token_iter()))
        self.assertEqual(('id:1234',), res)
        doc = dp.parse('id-1234')
        res = tuple(map(lambda x: x.norm, doc.token_iter()))
        self.assertEqual(('id-1234',), res)
        doc = dp.parse('an identifier: id-1234')
        res = tuple(map(lambda x: x.norm, doc.token_iter()))
        self.assertEqual(('an', 'identifier', ':', 'id-1234',), res)

    def test_detached_features(self):
        json_path = 'test-resources/detatch.json'
        dp = self.fac(
            'default_doc_parser',
            token_normalizer=self.fac('feature_no_filter_token_normalizer'))
        doc = dp.parse('I am a citizen of the United States of America.')
        feats = doc.token_iter()
        objs = []
        for f in feats:
            objs.append(f.asdict())
        if 0:
            with open(json_path, 'w') as f:
                json.dump(objs, f, indent=4)
        else:
            with open(json_path, 'r') as f:
                comps = json.load(f)
            self.assertEqual(comps, objs)

    def test_special_tok(self):
        sent = '<s> I am a citizen of the United States of America. </s>'
        self.assertEqual(('<', 's', '>', 'I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.', '<', '/s', '>'),
                         tuple(map(lambda t: t.norm, self._from_token_norm(
                             sent, 'nonorm_token_normalizer'))))
        dp = self.fac('special_doc_parser')
        doc = dp.parse(sent)
        self.assertEqual(('<s>', 'I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.', '</s>'),
                         tuple(map(lambda t: t.norm, doc.token_iter())))
