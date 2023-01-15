"""Produces matching scores.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Set, Dict, Any, Iterable, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys
from io import TextIOBase
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
    scores: Tuple[Dict[str, Tuple[Any]]] = field()
    """A tuple with each element having the results of the respective sentence
    pair in :obj:`.ScoreContext.sents`.  Each elemnt is a dictionary with the
    method are the keys with results as the values as output of the
    :class:`.ScoreMethod`.  This is created in :class:`.Scorer`.

    """
    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self) -> Iterable[Dict[str, Tuple[Any]]]:
        return iter(self.scores)

    def __getitem__(self, i: int) -> Dict[str, Tuple[Any]]:
        return self.scores[i]

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('scores:', depth, writer)
        self._write_iterable(self.scores, depth + 1, writer, include_index=True)


@dataclass
class ScoreContext(Dictable):
    """Input needed to create score(s) using :class:`.Scorer'.

    """
    pairs: Tuple[Tuple[FeatureSentence, FeatureSentence]] = field()
    """Sentence pairs to score (order matters for some scoring methods such as
    rouge).

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
    @abstractmethod
    def score(self, meth: str, context: ScoreContext) -> Any:
        """Score the sentences in ``context`` using method identifer ``meth``.

        :param meth: the identifer such as ``bleu``

        :param context: the context containing the data to score

        :return: the results, which are usually :class:`float` or
                 :class:`.Score`

        """
        pass

    def _tokenize(self, context: ScoreContext) -> \
            Iterable[Tuple[Tuple[str], Tuple[str]]]:
        s1: FeatureSentence
        s2: FeatureSentence
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
    def score(self, meth: str, context: ScoreContext) -> Iterable[float]:
        import nltk.translate.bleu_score as bleu
        for s1t, s2t in self._tokenize(context):
            yield bleu.sentence_bleu([s1t], s2t)


@dataclass
class RougeScoreMethod(ScoreMethod):
    """The ROUGE scoring method using the :mod:`rouge_score` package.

    """
    feature_tokenizer: bool = field(default=True)
    """Whether to use the :class:`.FeatureSentence` tokenization, otherwise use
    the :mod:`rouge_score` package.

    """
    @staticmethod
    def is_available() -> bool:
        try:
            import rouge_score
            return True
        except ModuleNotFoundError:
            return False

    def score(self, meth: str, context: ScoreContext) -> Iterable[Score]:
        from rouge_score import rouge_scorer

        class Tokenizer(object):
            @staticmethod
            def tokenize(sent: FeatureSentence) -> Tuple[str]:
                return sents[id(sent)]

        if self.feature_tokenizer:
            scorer = rouge_scorer.RougeScorer([meth], tokenizer=Tokenizer)
            s1: FeatureSentence
            s2: FeatureSentence
            pairs = zip(context.pairs, self._tokenize(context))
            for (s1, s2), (s1t, s2t) in pairs:
                sents = {id(s1): s1t, id(s2): s2t}
                res: Dict[str, Any] = scorer.score(s1, s2)
                yield Score(*res[meth])
        else:
            scorer = rouge_scorer.RougeScorer([meth])
            for s1t, s2t in context.pairs:
                res: Dict[str, Any] = scorer.score(
                    context.s1.text, context.s2.text)
                yield Score(*res[meth])


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
        by_meth: Dict[str, Tuple[Any]] = {}
        by_res: List[Dict[str, Any]] = []
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
            item_res: Dict[str, Any] = {}
            by_res.append(item_res)
            meth: str
            res_tup: Tuple[Any]
            for meth, res_tup in by_meth.items():
                item_res[meth] = res_tup[i]
        return ScoreSet(tuple(by_res))

    def __call__(self, context: ScoreContext) -> ScoreSet:
        """See :meth:`score`."""
        return self.score(context)
