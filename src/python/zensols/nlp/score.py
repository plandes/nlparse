"""Produces matching scores.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Set, Dict, Iterable, List
from dataclasses import dataclass, field
from abc import ABCMeta, ABC, abstractmethod
import logging
import sys
from io import TextIOBase
import nltk.translate.bleu_score as bleu
import numpy as np
from zensols.config import Dictable
from . import NLPError, TokenContainer

logger = logging.getLogger(__name__)


@dataclass
class Score(Dictable, metaclass=ABCMeta):
    """Individual scores returned from :class:`.ScoreMethod'.

    """
    def asrow(self, meth: str) -> Dict[str, float]:
        return {f'{meth}_{x[0]}': x[1] for x in self.asdict().items()}


@dataclass
class FloatScore(Score):
    """Float container.  This is needed to create the flat result container
    structure.  Object creation becomes less import since most clients will use
    :meth:`.ScoreSet.asnumpy`.

    """
    value: float = field()
    """The value of score."""

    def asrow(self, meth: str) -> Dict[str, float]:
        return {meth: self.value}


@dataclass
class HarmonicMeanScore(Score):
    """A score having a precision, recall and the harmonic mean of the two,
    F-score.'

    """
    precision: float = field()
    recall: float = field()
    f_score: float = field()


@dataclass
class ScoreSet(Dictable):
    """All scores returned from :class:`.Scorer'.

    """
    scores: Tuple[Dict[str, Tuple[Score]]] = field()
    """A tuple with each element having the results of the respective sentence
    pair in :obj:`.ScoreContext.sents`.  Each elemnt is a dictionary with the
    method are the keys with results as the values as output of the
    :class:`.ScoreMethod`.  This is created in :class:`.Scorer`.

    """
    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self) -> Iterable[Dict[str, Tuple[Score]]]:
        return iter(self.scores)

    def __getitem__(self, i: int) -> Dict[str, Tuple[Score]]:
        return self.scores[i]

    def as_numpy(self) -> Tuple[List[str], np.ndarray]:
        """Return the Numpy array with column descriptors of the scores.  Spacy
        depends on Numpy, so this package will always be availale.

        """
        cols: Set[str] = set()
        score: Dict[str, Score]
        rows: List[Dict[str, float]] = []
        for score in self.scores:
            row: Dict[str, float] = {}
            rows.append(row)
            meth: str
            result: Score
            for meth, result in score.items():
                rdat: Dict[str, float] = result.asrow(meth)
                row.update(rdat)
                cols.update(rdat.keys())
        cols: List[str] = sorted(cols)
        nd_rows: List[np.ndarray] = []
        for row in rows:
            nd_rows.append(np.array(tuple(map(row.get, cols))))
        arr = np.stack(nd_rows)
        return cols, arr

    def as_dataframe(self) -> 'pandas.DataFrame':
        """This gets data from :meth:`as_numpy` and returns it as a Pandas dataframe.

        :return: an instance of :class:`pandas.DataFrame` of the scores

        """
        import pandas as pd
        cols, arr = self.as_numpy()
        return pd.DataFrame(arr, columns=cols)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('scores:', depth, writer)
        self._write_iterable(self.scores, depth + 1, writer, include_index=True)


@dataclass
class ScoreContext(Dictable):
    """Input needed to create score(s) using :class:`.Scorer'.

    """
    pairs: Tuple[Tuple[TokenContainer, TokenContainer]] = field()
    """Sentence, span or document pairs to score (order matters for some scoring
    methods such as rouge).  For summarization use cases, the ordering of the
    sentence pairs should be ``(<source>, <summary>)``.

    """
    methods: Set[str] = field(default=None)
    """A set of strings, each indicating the :class:`.ScoreMethod` used to score
    :obj:`pairs`.

    """
    norm: bool = field(default=True)
    """Whether to use the normalized tokens, otherwise use the original text."""


@dataclass
class ScoreMethod(ABC):
    """An abstract base class for scoring methods (bleu, rouge, etc).

    """
    reverse_sents: bool = field(default=False)
    """Whether to reverse the order of the sentences."""

    @abstractmethod
    def _score(self, meth: str, context: ScoreContext) -> Score:
        """See :meth:`score`"""
        pass

    def score(self, meth: str, context: ScoreContext) -> Score:
        """Score the sentences in ``context`` using method identifer ``meth``.

        :param meth: the identifer such as ``bleu``

        :param context: the context containing the data to score

        :return: the results, which are usually :class:`float` or
                 :class:`.Score`

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'scoring meth: {meth}, ' +
                         f'reverse: {self.reverse_sents}')
        if self.reverse_sents:
            prev_pairs = context.pairs
            try:
                context.pairs = tuple(map(
                    lambda x: (x[1], x[0]), context.pairs))
                return tuple(self._score(meth, context))
            finally:
                context.pairs = prev_pairs
        else:
            return self._score(meth, context)

    def _tokenize(self, context: ScoreContext) -> \
            Iterable[Tuple[Tuple[str], Tuple[str]]]:
        s1: TokenContainer
        s2: TokenContainer
        for s1, s2 in context.pairs:
            s1t: Tuple[str]
            s2t: Tuple[str]
            if context.norm:
                s1t = tuple(map(lambda t: t.norm, s1.token_iter()))
                s2t = tuple(map(lambda t: t.norm, s2.token_iter()))
            else:
                s1t = tuple(map(lambda t: t.text, s1.token_iter()))
                s2t = tuple(map(lambda t: t.text, s2.token_iter()))
            yield (s1t, s2t)


@dataclass
class BleuScoreMethod(ScoreMethod):
    """The BLEU scoring method using the :mod:`nltk` package.

    """
    smoothing_function: bleu.SmoothingFunction = field(default=None)
    """This is an implementation of the smoothing techniques for segment-level
    BLEU scores that was presented in Boxing Chen and Collin Cherry (2014) A
    Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU. In
    WMT14.  http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf

    """
    weights: Tuple[float, ...] = field(default=(0.25, 0.25, 0.25, 0.25))
    """Weights for each n-gram.  For example: a tuple of float weights for
    unigrams, bigrams, trigrams and so on can be given: ``weights = (0.1, 0.3,
    0.5, 0.1)``.

    """
    def _score(self, meth: str, context: ScoreContext) -> Iterable[float]:
        for s1t, s2t in self._tokenize(context):
            val: float = bleu.sentence_bleu(
                [s1t], s2t,
                weights=self.weights,
                smoothing_function=self.smoothing_function)
            yield FloatScore(val)


@dataclass
class RougeScoreMethod(ScoreMethod):
    """The ROUGE scoring method using the :mod:`rouge_score` package.

    """
    feature_tokenizer: bool = field(default=True)
    """Whether to use the :class:`.TokenContainer` tokenization, otherwise use
    the :mod:`rouge_score` package.

    """
    @staticmethod
    def is_available() -> bool:
        try:
            import rouge_score
            return True
        except ModuleNotFoundError:
            return False

    def _score(self, meth: str, context: ScoreContext) -> \
            Iterable[HarmonicMeanScore]:
        from rouge_score import rouge_scorer

        class Tokenizer(object):
            @staticmethod
            def tokenize(sent: TokenContainer) -> Tuple[str]:
                return sents[id(sent)]

        if self.feature_tokenizer:
            scorer = rouge_scorer.RougeScorer([meth], tokenizer=Tokenizer)
            s1: TokenContainer
            s2: TokenContainer
            pairs = zip(context.pairs, self._tokenize(context))
            for (s1, s2), (s1t, s2t) in pairs:
                sents = {id(s1): s1t, id(s2): s2t}
                res: Dict[str, Score] = scorer.score(s1, s2)
                yield HarmonicMeanScore(*res[meth])
        else:
            scorer = rouge_scorer.RougeScorer([meth])
            for s1t, s2t in context.pairs:
                res: Dict[str, Score] = scorer.score(
                    context.s1.text, context.s2.text)
                yield HarmonicMeanScore(*res[meth])


@dataclass
class Scorer(object):
    """A class that scores sentences using a set of registered methods
    (:ob:`methods`).

    """
    methods: Dict[str, ScoreMethod] = field(default=None)
    """The registered scoring methods availale, which are accessed from
    :obj:`.ScoreContext.meth`.

    """
    def score(self, context: ScoreContext) -> ScoreSet:
        """Score the sentences in ``context``.

        :param context: the context containing the data to score

        :return: the results for each method indicated in ``context``

        """
        by_meth: Dict[str, Tuple[Score]] = {}
        by_res: List[Dict[str, Score]] = []
        meths: Iterable[str] = context.methods
        if meths is None:
            meths = self.methods.keys()
        meth: str
        for meth in meths:
            smeth: ScoreMethod = self.methods.get(meth)
            if smeth is None:
                raise NLPError(f"No scoring method: '{meth}'")
            by_meth[meth] = tuple(smeth.score(meth, context))
        for i in range(len(context.pairs)):
            item_res: Dict[str, Score] = {}
            by_res.append(item_res)
            meth: str
            res_tup: Tuple[Score]
            for meth, res_tup in by_meth.items():
                item_res[meth] = res_tup[i]
        return ScoreSet(scores=tuple(by_res))

    def __call__(self, context: ScoreContext) -> ScoreSet:
        """See :meth:`score`."""
        return self.score(context)
