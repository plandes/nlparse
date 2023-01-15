"""Produces matching scores.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Set, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from zensols.config import Dictable
from . import NLPError, FeatureSentence


@dataclass
class Score(Dictable):
    """Individual scores returned from :class:`.ScoreMethod'.

    """
    precision: float = field()
    recall: float = field()
    f_score: float = field()


@dataclass
class ScoreSet(Dictable):
    """All scores returned from :class:`.Scorer'.

    """
    scores: Dict[str, Any] = field()
    """The method are the keys with results as the values as output of the
    :class:`.ScoreMethod`.  This is created in :class:`.Scorer`.

    """


@dataclass
class ScoreContext(Dictable):
    """Input needed to create score(s) using :class:`.Scorer'.

    """
    s1: FeatureSentence = field()
    """The first sentence to be compared (order matters for some scoring methods
    such as rouge).

    """
    s2: FeatureSentence = field()
    """The first sentence to be compared (order matters for some scoring methods
    such as rouge).

    """
    methods: Set[str] = field()
    """A set of strings, each indicating the :class:`.ScoreMethod` used to score
    :obj:`s1` with :obj:`s2`.

    """
    norm: bool = field(default=True)
    """Whether to use the normalized tokens, otherwise use the original text."""


@dataclass
class ScoreMethod(ABC):
    """An abstract base class for scoring methods (bleu, rouge, etc).

    """
    @abstractmethod
    def score(self, meth: str, context: ScoreContext) -> Any:
        """Score the sentences in ``context`` using method identifer ``meth``.

        :param meth: the identifer such as ``bleu``

        :param context: the context containing the data to score

        :return: the results, which are usually :class:`float` or
                 :class:`.Score`

        """
        pass

    def _tokenize(self, context: ScoreContext) -> Tuple[Tuple[str], Tuple[str]]:
        s1t: Tuple[str]
        s2t: Tuple[str]
        if context.norm:
            s1t = tuple(map(lambda t: t.norm, context.s1.token_iter()))
            s2t = tuple(map(lambda t: t.norm, context.s2.token_iter()))
        else:
            s1t = tuple(map(lambda t: t.text, context.s1.token_iter()))
            s2t = tuple(map(lambda t: t.text, context.s2.token_iter()))
        return s1t, s2t


@dataclass
class BleuScoreMethod(ScoreMethod):
    """The BLEU scoring method using the :mod:`nltk` package.

    """
    def score(self, meth: str, context: ScoreContext) -> float:
        import nltk.translate.bleu_score as bleu
        s1t, s2t = self._tokenize(context)
        return bleu.sentence_bleu([s1t], s2t)


@dataclass
class RougeScoreMethod(ScoreMethod):
    """The ROUGE scoring method using the :mod:`rouge_score` package.

    """
    feature_tokenizer: bool = field(default=True)
    """Whether to use the :class:`.FeatureSentence` tokenization, otherwise use
    the :mod:`rouge_score` package.

    """
    def score(self, meth: str, context: ScoreContext) -> float:
        from rouge_score import rouge_scorer

        class Tokenizer(object):
            @staticmethod
            def tokenize(sent: FeatureSentence) -> Tuple[str]:
                return sents[id(sent)]

        if self.feature_tokenizer:
            s1t, s2t = self._tokenize(context)
            sents = {id(context.s1): s1t, id(context.s2): s2t}
            scorer = rouge_scorer.RougeScorer([meth], tokenizer=Tokenizer)
            res: Dict[str, Any] = scorer.score(context.s1, context.s2)
        else:
            scorer = rouge_scorer.RougeScorer([meth])
            res: Dict[str, Any] = scorer.score(context.s1.text, context.s2.text)
        return Score(*res[meth])


@dataclass
class Scorer(object):
    """A class that scores sentences using a set of registered methods
    (:ob:`methods`).

    """
    methods: Dict[str, ScoreMethod] = field()
    """The registered scoring methods availale, which are accessed from
    :obj:`.ScoreContext.meth`.

    """
    def score(self, context: ScoreContext) -> ScoreSet:
        """Score the sentences in ``context``.

        :param context: the context containing the data to score

        :return: the results for each method indicated in ``context``

        """
        res: Dict[str, Any] = {}
        meth: str
        for meth in context.methods:
            smeth: ScoreMethod = self.methods.get(meth)
            if smeth is None:
                raise NLPError(f"No scoring method: '{meth}'")
            res[meth] = smeth.score(meth, context)
        return ScoreSet(res)
