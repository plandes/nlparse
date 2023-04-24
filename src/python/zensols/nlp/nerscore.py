"""Wraps the `SemEval-2013 Task 9.1`_ NER evaluation API as a
:class:`~zensols.nlp.score.ScoreMethod`.

From the `David Batista`_ blog post:

  The SemEval’13 introduced four different ways to measure
  precision/recall/f1-score results based on the metrics defined by MUC:

    * *Strict*: exact boundary surface string match and entity type

    * *Exact*: exact boundary match over the surface string, regardless of the
      type

    * *Partial*: partial boundary match over the surface string, regardless of
      the type

    * *Type*: some overlap between the system tagged entity and the gold
      annotation is required

  Each of these ways to measure the performance accounts for correct, incorrect,
  partial, missed and spurious in different ways. Let’s look in detail and see
  how each of the metrics defined by MUC falls into each of the scenarios
  described above.


:see: `SemEval-2013 Task 9.1 <https://web.archive.org/web/20150131105418/https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/semeval_2013-task-9_1-evaluation-metrics.pdf>`_

:see: `David Batista <http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/>`_

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Tuple, Dict, Set, List, Optional, Any, Iterable, ClassVar, Type
)
from dataclasses import dataclass, field, fields
import numpy as np
from zensols.nlp import TokenContainer, FeatureSpan
from zensols.nlp.score import (
    Score, ErrorScore, ScoreMethod, ScoreContext, HarmonicMeanScore
)


@dataclass
class SemEvalHarmonicMeanScore(HarmonicMeanScore):
    """A harmonic mean score with the additional SemEval computed scores (see
    module :mod:`zensols.nlp.nerscore` docs).

    """
    NAN_INSTANCE: ClassVar[SemEvalHarmonicMeanScore] = None

    correct: int = field()
    """The number of correct (COR): both are the same."""

    incorrect: int = field()
    """The number of incorrect (INC): the output of a system and the golden
    annotation don’t match.

    """
    partial: int = field()
    """The number of partial (PAR): system and the golden annotation are
    somewhat “similar” but not the same.

    """
    missed: int = field()
    """The number of missed (MIS): a golden annotation is not captured by a
    system."""

    spurious: int = field()
    """The number of spurious (SPU): system produces a response which does not
    exist in the golden annotation.

    """
    possible: int = field()
    actual: int = field()


SemEvalHarmonicMeanScore.NAN_INSTANCE = SemEvalHarmonicMeanScore(
    *[np.nan] * 10)


@dataclass
class SemEvalScore(Score):
    """Contains all four harmonic mean SemEval scores (see module
    :mod:`zensols.nlp.nerscore` docs).  This score has four harmonic means
    providing various levels of accuracy.

    """
    NAN_INSTANCE: ClassVar[SemEvalScore] = None

    strict: SemEvalHarmonicMeanScore = field()
    """Exact boundary surface string match and entity type."""

    exact: SemEvalHarmonicMeanScore = field()
    """Exact boundary match over the surface string, regardless of the type."""

    partial: SemEvalHarmonicMeanScore = field()
    """Partial boundary match over the surface string, regardless of the type.

    """
    ent_type: SemEvalHarmonicMeanScore = field()
    """Some overlap between the system tagged entity and the gold annotation is
    required.

    """
    def asrow(self, meth: str) -> Dict[str, float]:
        row: Dict[str, Any] = {}
        f: field
        for f in fields(self):
            score: Score = getattr(self, f.name)
            row.update(score.asrow(f'{meth}_{f.name}'))
        return row


SemEvalScore.NAN_INSTANCE = SemEvalScore(
    partial=SemEvalHarmonicMeanScore.NAN_INSTANCE,
    strict=SemEvalHarmonicMeanScore.NAN_INSTANCE,
    exact=SemEvalHarmonicMeanScore.NAN_INSTANCE,
    ent_type=SemEvalHarmonicMeanScore.NAN_INSTANCE)


@dataclass
class SemEvalScoreMethod(ScoreMethod):
    """A Semeval-2013 Task 9.1 scor (see module :mod:`zensols.nlp.nerscore`
    docs).  This score has four harmonic means providing various levels of
    accuracy.  Sentence pairs are ordered as ``(<gold>, <prediction>)``.

    """
    labels: Optional[Set[str]] = field(default=None)
    """The NER labels on which to evaluate.  If not provided, text is evaluated
    under a (stubbed tag) label.

    """
    @classmethod
    def _get_external_modules(cls: Type) -> Tuple[str, ...]:
        return ('nervaluate',)

    def _score_pair(self, gold: TokenContainer, pred: TokenContainer) -> \
            SemEvalScore:
        from nervaluate import Evaluator

        def nolab(c: TokenContainer, label: str) -> Tuple[Dict[str, Any], ...]:
            return tuple(map(
                lambda t: dict(label=label, start=t.lexspan.begin,
                               end=t.lexspan.end),
                c.token_iter()))

        def withlab(c: TokenContainer) -> Tuple[Dict[str, Any]]:
            ent_set: List[Tuple[Dict[str, Any], ...], ...] = []
            ent: FeatureSpan
            for ent in c.entities:
                ents: Tuple[Dict[str, Any], ...] = tuple(map(
                    lambda t: dict(label=t.ent_, start=t.lexspan.begin,
                                   end=t.lexspan.end), ent))
                ent_set.append(ents)
            return tuple(ent_set)

        tags: Tuple[str, ...]
        gold_ents: Tuple[Dict[str, Any], ...]
        pred_ents: Tuple[Dict[str, Any], ...]
        if self.labels is None:
            label: str = '_'
            gold_ents, pred_ents = nolab(gold, label), nolab(pred, label)
            gold_ents, pred_ents = (gold_ents,), (pred_ents,)
            tags = (label,)
        else:
            gold_ents, pred_ents = withlab(gold), withlab(pred)
            tags = tuple(self.labels)

        evaluator = Evaluator(gold_ents, pred_ents, tags=tags)
        res: Dict[str, Any] = evaluator.evaluate()[0]
        hscores: Dict[str, SemEvalHarmonicMeanScore] = {}
        k: str
        hdat: Dict[str, float]
        for k, hdat in res.items():
            hdat['f_score'] = hdat.pop('f1')
            hscores[k] = (SemEvalHarmonicMeanScore(**hdat))
        return SemEvalScore(**hscores)

    def _score(self, meth: str, context: ScoreContext) -> \
            Iterable[SemEvalScore]:
        gold: TokenContainer
        pred: TokenContainer
        for gold, pred in context.pairs:
            try:
                yield self._score_pair(gold, pred)
            except Exception as e:
                yield ErrorScore(meth, e, SemEvalScore.NAN_INSTANCE)
