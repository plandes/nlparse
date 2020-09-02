import logging
import unittest
import json
from zensols.config import ImportConfigFactory
from zensols.nlp import (
    LanguageResource,
    DocUtil,
)
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
        self.lr = self.fac.instance('default_langres')

    def test_parse(self):
        lr = self.lr
        self.assertTrue(isinstance(lr, LanguageResource))
        doc = lr.parse('Dan throws the ball.')
        res = DocUtil.to_json(doc)
        with open(self.config.parse_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))

    def test_feature(self):
        tnfac = ImportConfigFactory(self.config, shared=False)
        tn = tnfac.instance('default_token_normalizer')
        lr = self.fac.instance('default_langres', token_normalizer=tn)
        doc = lr.parse('Dan throws the ball.')
        self.assertEqual('MapTokenNormalizer: embed=True', str(tn))

        res = tuple(map(lambda x: x.string_features, lr.features(doc)))
        with open(self.config.feature_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))

        tn = tnfac.instance('nonorm_token_normalizer')
        lr = self.fac.instance('default_langres', token_normalizer=tn)
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def test_map(self):
        tnfac = ImportConfigFactory(self.config)
        doc = self.lr.parse('I am a citizen of the United States of America.')
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_token_normalizer'))))
        self.assertEqual(('am', 'citizen', 'of', 'the United States of America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_pron_token_normalizer'))))
        self.assertEqual(('i', 'am', 'a', 'citizen', 'of', 'the united states of america', '.'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_lower_token_normalizer'))))
        self.assertEqual(('citizen', 'United', 'States', 'America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_embed_token_normalizer'))))
        self.assertEqual(('citizen', 'the_united_states_of_america'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_subs_token_normalizer'))))

    def test_disable(self):
        lr = self.lr
        dis_lr = self.fac.instance('disable_tagger_langres')
        doc = lr.parse('Dan throws the ball.')
        tags = tuple(map(lambda t: t.tag_, doc))
        self.assertEqual(('NNP', 'VBZ', 'DT', 'NN', '.'), tags)
        self.assertEqual('tagger parser'.split(), dis_lr.disable_components)
        doc_dis = dis_lr.parse('Dan throws the ball.')
        no_tags = tuple(map(lambda t: t.tag_, doc_dis))
        self.assertEqual(('', '', '', '', ''), no_tags)

    def test_filter_features(self):
        tnfac = ImportConfigFactory(self.config)
        lr = self.fac.instance('default_langres', token_normalizer=tnfac.instance('feature_no_filter_token_normalizer'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        self.assertEqual(('I', 'am', 'a', 'citizen', 'of', 'the United States of America', '.'),
                         tuple(map(lambda f: f.norm, feats)))

        lr = self.fac.instance('default_langres', token_normalizer=tnfac.instance('feature_default_filter_token_normalizer'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        self.assertEqual(('I', 'am', 'citizen', 'of', 'the United States of America'),
                         tuple(map(lambda f: f.norm, feats)))

        lr = self.fac.instance('default_langres', token_normalizer=tnfac.instance('feature_stop_filter_token_normalizer'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(map(lambda f: f.norm, feats)))

    def test_space(self):
        tnfac = ImportConfigFactory(self.config)
        tn = tnfac.instance('nonorm_token_normalizer')
        lr = self.fac.instance('default_langres', token_normalizer=tn)
        doc = lr.parse('''Dan throws
the ball.''')
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('Dan', 'throws', '\n', 'the', 'ball', '.'), res)

        tn = tnfac.instance('map_filter_space_token_normalizer')
        lr = self.fac.instance('default_langres', token_normalizer=tn)
        doc = lr.parse('''Dan throws
the ball.''')
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def test_tok_boundaries(self):
        tnfac = ImportConfigFactory(self.config)
        tn = tnfac.instance('nonorm_token_normalizer')
        lr = self.fac.instance('default_langres', token_normalizer=tn)
        doc = lr.parse('id:1234')
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('id:1234',), res)
        doc = lr.parse('id-1234')
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('id-1234',), res)
        doc = lr.parse('an identifier: id-1234')
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('an', 'identifier', ':', 'id-1234',), res)

    def test_detached_features(self):
        json_path = 'test-resources/detatch.json'
        tnfac = ImportConfigFactory(self.config)
        lr = self.fac.instance('default_langres', token_normalizer=tnfac.instance('feature_no_filter_token_normalizer'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        objs = []
        for f in feats:
            objs.append(f.detach().to_dict())
        if 0:
            with open(json_path, 'w') as f:
                json.dump(objs, f, indent=4)
        else:
            with open(json_path, 'r') as f:
                comps = json.load(f)
            self.assertEqual(comps, objs)

    def test_special_tok(self):
        tnfac = ImportConfigFactory(self.config)
        txt = '<s> I am a citizen of the United States of America. </s>'
        doc = self.lr.parse(txt)
        self.assertEqual(('<', 's', '>', 'I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.', '<', '/s', '>'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('nonorm_token_normalizer'))))
        lr = self.fac.instance('special_langres')
        doc = lr.parse(txt)
        self.assertEqual(('<s>', 'I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.', '</s>'),
                         tuple(lr.normalized_tokens(doc)))
