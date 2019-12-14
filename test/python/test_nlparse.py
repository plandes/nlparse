import logging
import unittest
import json
from zensols.actioncli import ExtendedInterpolationConfig
from zensols.nlp import (
    LanguageResourceFactory,
    LanguageResource,
    DocUtil,
    TokenNormalizerFactory,
)

logger = logging.getLogger(__name__)


def rec_sort(x):
    if isinstance(x, list) or isinstance(x, tuple):
        x = sorted(list(map(rec_sort, x)))
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = rec_sort(v)
        x = sorted(x.items())
    return x


class AppConfig(ExtendedInterpolationConfig):
    SECTION = 'test'

    def __init__(self):
        super(AppConfig, self).__init__(
            config_file='test-resources/nlparse.conf', default_expect=True)

    @property
    def parse_path(self):
        return self.get_option_path('parse_test', self.SECTION)

    @property
    def feature_path(self):
        return self.get_option_path('feature_test', self.SECTION)


class TestParse(unittest.TestCase):
    def setUp(self):
        self.maxDiff = 999999
        self.config = AppConfig()
        self.fac = LanguageResourceFactory(self.config)
        self.lr = self.fac.instance()

    def test_parse(self):
        lr = self.lr
        self.assertTrue(isinstance(lr, LanguageResource))
        doc = lr.parse('Dan throws the ball.')
        res = DocUtil.to_json(doc)
        with open(self.config.parse_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))

    def test_feature(self):
        #lr = self.lr
        tnfac = TokenNormalizerFactory(self.config)
        tn = tnfac.instance()
        lr = self.fac.instance(token_normalizer=tn)
        doc = lr.parse('Dan throws the ball.')
        self.assertEqual('TokenNormalizer: embed=True, normalize: True remove first stop: False', str(tn))

        res = tuple(map(lambda x: x.string_features, lr.features(doc)))
        with open(self.config.feature_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))

        tn = tnfac.instance('nonorm')
        lr = self.fac.instance(token_normalizer=tn)
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def test_map(self):
        tnfac = TokenNormalizerFactory(self.config)
        doc = self.lr.parse('I am a citizen of the United States of America.')
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter'))))
        self.assertEqual(('am', 'citizen', 'of', 'the United States of America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_pron'))))
        self.assertEqual(('i', 'am', 'a', 'citizen', 'of', 'the united states of america', '.'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_lower'))))
        self.assertEqual(('citizen', 'United', 'States', 'America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_embed'))))
        self.assertEqual(('citizen', 'the_united_states_of_america'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_subs'))))

    def test_disable(self):
        lr = self.lr
        dis_lr = self.fac.instance('disable_tagger')
        doc = lr.parse('Dan throws the ball.')
        tags = tuple(map(lambda t: t.tag_, doc))
        self.assertEqual(('NNP', 'VBZ', 'DT', 'NN', '.'), tags)
        self.assertEqual('tagger parser'.split(), dis_lr.disable_components)
        doc_dis = dis_lr.parse('Dan throws the ball.')
        no_tags = tuple(map(lambda t: t.tag_, doc_dis))
        self.assertEqual(('', '', '', '', ''), no_tags)

    def test_filter_features(self):
        tnfac = TokenNormalizerFactory(self.config)
        lr = self.fac.instance(token_normalizer=tnfac.instance('feature_no_filter'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        self.assertEqual(('I', 'am', 'a', 'citizen', 'of', 'the United States of America', '.'),
                         tuple(map(lambda f: f.norm, feats)))

        lr = self.fac.instance(token_normalizer=tnfac.instance('feature_default_filter'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        self.assertEqual(('I', 'am', 'citizen', 'of', 'the United States of America'),
                         tuple(map(lambda f: f.norm, feats)))

        lr = self.fac.instance(token_normalizer=tnfac.instance('feature_stop_filter'))
        doc = self.lr.parse('I am a citizen of the United States of America.')
        feats = lr.features(doc)
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(map(lambda f: f.norm, feats)))

    def test_space(self):
        tnfac = TokenNormalizerFactory(self.config)
        tn = tnfac.instance('nonorm')
        lr = self.fac.instance(token_normalizer=tn)
        doc = lr.parse('''Dan throws
the ball.''', normalize=False)
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('Dan', 'throws', '\n', 'the', 'ball', '.'), res)

        tn = tnfac.instance('map_filter_space')
        lr = self.fac.instance(token_normalizer=tn)
        doc = lr.parse('''Dan throws
the ball.''', normalize=False)
        res = tuple(map(lambda x: x.norm, lr.features(doc)))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def test_tok_boundaries(self):
        tnfac = TokenNormalizerFactory(self.config)
        tn = tnfac.instance('nonorm')
        lr = self.fac.instance(token_normalizer=tn)
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
        tnfac = TokenNormalizerFactory(self.config)
        lr = self.fac.instance(token_normalizer=tnfac.instance('feature_no_filter'))
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
