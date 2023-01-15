import unittest
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp.score import ScoreContext, RougeScoreMethod


class TestScore(unittest.TestCase):
    def setUp(self):
        config = ImportIniConfig('test-resources/score.conf')
        self.fac = ImportConfigFactory(config)
        self.doc_parser = self.fac('doc_parser')
        self.scorer = self.fac('nlp_sent_scorer')

    def test_score(self):
        s1 = self.doc_parser('Dan threw the ball.')
        s2 = self.doc_parser('The boy threw the ball.')
        s3 = self.doc_parser('The boy threw the ball and then ran.')
        meths = {'bleu'}
        rouge_avail = RougeScoreMethod.is_available()
        if rouge_avail:
            meths.add('rouge1')
        res = self.scorer(ScoreContext([[s1, s2], [s2, s3], [s3, s2]], meths))
        self.assertEqual(3, len(res))

        r1 = res[0]
        self.assertTrue(isinstance(r1, dict))
        self.assertEqual(len(meths), len(r1))
        bleu = r1['bleu']
        self.assertEqual(0.51, round(bleu, 2))
        if rouge_avail:
            self.assertEqual(0.73, round(r1['rouge1'].f_score, 2))

        self.assertEqual(0.47, round(res[1]['bleu'], 2))
        if rouge_avail:
            self.assertEqual(0.8, round(res[1]['rouge1'].f_score, 2))

        self.assertEqual(0.48, round(res[2]['bleu'], 2))
        if rouge_avail:
            self.assertEqual(0.8, round(res[1]['rouge1'].f_score, 2))
