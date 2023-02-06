from typing import Iterable
from dataclasses import dataclass, field
import unittest
import pickle
from io import BytesIO
import numpy as np
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import NLPError
from zensols.nlp.score import (
    ScoreContext, RougeScoreMethod, ScoreMethod, ScoreSet, ScoreResult
)


@dataclass
class ErrorScoreMethod(ScoreMethod):
    raise_ex: bool = field(repr=False, default=True)

    def _score(self, meth: str, context: ScoreContext) -> Iterable[float]:
        if self.raise_ex:
            raise ValueError('Artificial error')
        else:
            from zensols.nlp.score import FloatScore
            for _ in context.pairs:
                yield FloatScore(-1)


class TestScore(unittest.TestCase):
    def _init(self, name: str):
        config = ImportIniConfig(f'test-resources/{name}.conf')
        self.fac = ImportConfigFactory(config)
        self.doc_parser = self.fac('doc_parser')
        self.scorer = self.fac('nlp_scorer')

    def _ser_test(self, ss: ScoreSet):
        bio = BytesIO()
        pickle.dump(ss, bio)
        bio.seek(0)
        ss2 = pickle.load(bio)
        self.assertEqual(ss, ss2)

    def test_score(self):
        self._init('score')
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
        self.assertTrue(isinstance(r1, ScoreResult))
        self.assertEqual(len(meths), len(r1))
        bleu = r1['bleu']
        self.assertEqual(0.51, round(bleu.value, 2))
        if rouge_avail:
            self.assertEqual(0.73, round(r1['rouge1'].f_score, 2))

        self.assertEqual(0.47, round(res[1]['bleu'].value, 2))
        if rouge_avail:
            self.assertEqual(0.8, round(res[1]['rouge1'].f_score, 2))

        self.assertEqual(0.48, round(res[2]['bleu'].value, 2))
        if rouge_avail:
            self.assertEqual(0.8, round(res[1]['rouge1'].f_score, 2))

        self.assertEqual(0.47, round(res[1]['bleu'].value, 2))

        self._ser_test(res)

    def test_score_err(self):
        self._init('score-error')
        s1 = self.doc_parser('Dan threw the ball.')
        s2 = self.doc_parser('The boy threw the ball.')
        s3 = self.doc_parser('The boy threw the ball and then ran.')
        res = self.scorer(ScoreContext([[s1, s2], [s2, s3]]))
        r1 = res[0]
        bleu = r1['bleu']
        self.assertEqual(0.51, round(bleu.value, 2))
        err = r1['err']
        self.assertEqual('Artificial error', str(err.exception))
        non_err = r1['nonerr']
        self.assertEqual(-1, non_err.value)
        cols, arr = res.as_numpy()
        if 0:
            print(arr[:, :4])
            print(res.as_dataframe().to_csv('/d/a.csv'))
            return
        self.assertEqual(['bleu', 'err', 'nonerr'], cols)
        should = np.float64(np.array(
            [[0.51, np.nan, -1.],
             [0.47, np.nan, -1.]]))
        self.assertTrue(np.array_equal(should, arr.round(2), equal_nan=True))
        self._ser_test(res)

    def test_correlation_id(self):
        self._init('score')
        s1 = self.doc_parser('Dan threw the ball.')
        s2 = self.doc_parser('The boy threw the ball.')
        s3 = self.doc_parser('The boy threw the ball and then ran.')
        with self.assertRaises(NLPError):
            self.scorer(ScoreContext(
                [[s1, s2], [s2, s3]],
                methods={'bleu'},
                correlation_ids='1 2 3'.split()))
        res = self.scorer(ScoreContext(
            [[s1, s2], [s2, s3]],
            methods={'bleu'},
            correlation_ids='1 2'.split(),
        ))
        self.assertEqual(2, len(res))
        self.assertEqual(0.51, round(res[0]['bleu'].value, 2))
        self.assertEqual('1', res[0].correlation_id)
        self.assertEqual('2', res[1].correlation_id)
        cols, arr = res.as_numpy()
        self.assertEqual(['bleu', 'id'], cols)
        has_pandas = False
        try:
            import pandas
            has_pandas = True
        except Exception:
            pass
        if has_pandas:
            df = res.as_dataframe()
            self.assertEqual(['bleu', 'id'], df.columns.tolist())
            self.assertEqual([0.51, 0.47], df.iloc[:, 0].round(2).tolist())
