"""Produces matching scores.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Tuple, Set, Dict, Iterable, List, ClassVar, Union, Optional, Type
)
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import sys
from io import TextIOBase
import nltk.translate.bleu_score as bleu
import numpy as np
from zensols.util import PackageRequirement, PackageManager, PackageError
from zensols.introspect import ClassImporter
from zensols.config import Dictable
from zensols.persist import persisted
from zensols.nlp import TokenContainer
from . import NLPError

logger = logging.getLogger(__name__)


class ScorerError(NLPError):
    """Raised for any scoring errors (this module)."""
    pass


@dataclass
class Score(Dictable, metaclass=ABCMeta):
    """Individual scores returned from :class:`.ScoreMethod`.

    """
    def asrow(self, meth: str) -> Dict[str, float]:
        return {f'{meth}_{x[0]}': x[1] for x in self.asdict().items()}


@dataclass(eq=False)
class ErrorScore(Score):
    """A replacement instance when scoring fails from a raised exception.

    """
    method: str = field(repr=False)
    """The method of the :class:`.ScoreMethod` that raised the exception."""

    exception: Exception = field()
    """The exception that was raised."""

    replace_score: Score = field(default=None)
    """The score to use in place of this score.  Otherwise :meth:`asrow` return
    a single :obj:`numpy.nan` like :class:`.FloatScore`.

    """
    def asrow(self, meth: str) -> Dict[str, float]:
        if self.replace_score is not None:
            return self.replace_score.asrow(self.method)
        return {self.method: np.nan}

    def __eq___(self, other) -> bool:
        return self.method == other.method and \
            str(self.exception) == str(other.exeption)


@dataclass
class FloatScore(Score):
    """Float container.  This is needed to create the flat result container
    structure.  Object creation becomes less import since most clients will use
    :meth:`.ScoreSet.asnumpy`.

    """
    NAN_INSTANCE: ClassVar[FloatScore] = None
    """Used to add to ErrorScore for harmonic means replacements.

    """

    value: float = field()
    """The value of score."""

    def asrow(self, meth: str) -> Dict[str, float]:
        return {meth: self.value}


FloatScore.NAN_INSTANCE = FloatScore(np.nan)


@dataclass
class HarmonicMeanScore(Score):
    """A score having a precision, recall and the harmonic mean of the two,
    F-score.'

    """
    NAN_INSTANCE: ClassVar[HarmonicMeanScore] = None
    """Used to add to ErrorScore for harmonic means replacements.

    """
    precision: float = field()
    recall: float = field()
    f_score: float = field()


HarmonicMeanScore.NAN_INSTANCE = HarmonicMeanScore(np.nan, np.nan, np.nan)


@dataclass
class ScoreResult(Dictable):
    """A result of scores created by a :class:`.ScoreMethod`.

    """
    scores: Dict[str, Tuple[Score, ...]] = field()
    """The scores by method name."""

    correlation_id: Optional[str] = field(default=None)
    """An ID for correlating back to the :class:`.TokenContainer`."""

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, k: str) -> Dict[str, Tuple[Score, ...]]:
        return self.scores[k]

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct = super().asdict()
        del dct['correlation_id']
        if self.correlation_id is None:
            self._write_dict(dct, depth, writer)
        else:
            self._write_line(f'correlation ID: {self.correlation_id}',
                             depth, writer)
            self._write_dict(dct, depth + 1, writer)


@dataclass
class ScoreSet(Dictable):
    """All scores returned from :class:`.Scorer'.

    """
    results: Tuple[ScoreResult, ...] = field()
    """A tuple with each element having the results of the respective sentence
    pair in :obj:`.ScoreContext.sents`.  Each elemnt is a dictionary with the
    method are the keys with results as the values as output of the
    :class:`.ScoreMethod`.  This is created in :class:`.Scorer`.

    """
    correlation_id_col: str = field(default='id')
    """The column name for the :obj:`.ScoreResult.correlation_id` added to Numpy
    arrays and Pandas dataframes.  If ``None``, then the correlation IDS are
    used as the index.

    """
    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterable[Dict[str, Tuple[Score, ...]]]:
        return iter(self.results)

    def __getitem__(self, i: int) -> Dict[str, Tuple[Score, ...]]:
        return self.results[i]

    @property
    def has_correlation_id(self) -> bool:
        """Whether the results have correlation IDs."""
        return len(self.results) > 0 and \
            self.results[0].correlation_id is not None

    def as_numpy(self, add_correlation: bool = True) -> \
            Tuple[List[str], np.ndarray]:
        """Return the Numpy array with column descriptors of the results.  Spacy
        depends on Numpy, so this package will always be availale.

        :param add_correlation: whether to add the correlation ID (if there is
                                one), using :obj:`correlation_id_col`

        """
        cols: Set[str] = set()
        rows: List[Dict[str, float]] = []
        result: ScoreResult
        for result in self.results:
            row: Dict[str, float] = {}
            rows.append(row)
            meth: str
            for meth, result in result.scores.items():
                rdat: Dict[str, float] = result.asrow(meth)
                row.update(rdat)
                cols.update(rdat.keys())
        cols: List[str] = sorted(cols)
        nd_rows: List[np.ndarray] = []
        for row in rows:
            nd_rows.append(np.array(tuple(map(row.get, cols))))
        arr = np.stack(nd_rows)
        if add_correlation and self.has_correlation_id:
            ids = np.array(tuple(map(lambda r: r.correlation_id, self.results)))
            ids = np.expand_dims(ids, 1)
            arr = np.append(arr, ids, axis=1)
            cols.append(self.correlation_id_col)
        return cols, arr

    def as_dataframe(self, add_correlation: bool = True) -> 'pandas.DataFrame':
        """This gets data from :meth:`as_numpy` and returns it as a Pandas
        dataframe.

        :param add_correlation: whether to add the correlation ID (if there is
                                one), using :obj:`correlation_id_col`

        :return: an instance of :class:`pandas.DataFrame` of the results

        """
        import pandas as pd
        cols, arr = self.as_numpy(add_correlation=False)
        df = pd.DataFrame(arr, columns=cols)
        if add_correlation and self.has_correlation_id:
            # add as a dataframe, otherwise string correlation IDs cast the
            # numpy array to a string
            cid: str = self.correlation_id_col
            cids: Tuple[Union[str, int]] = tuple(
                map(lambda r: r.correlation_id, self.results))
            if cid is None:
                df.index = cids
            else:
                cols: List[str] = df.columns.tolist()
                df[cid] = cids
                cols.insert(0, cid)
                df = df[cols]
        return df

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('results:', depth, writer)
        self._write_iterable(self.results, depth + 1, writer)


@dataclass
class ScoreContext(Dictable):
    """Input needed to create score(s) using :class:`.Scorer`.

    """
    pairs: Tuple[Tuple[TokenContainer, TokenContainer]] = field()
    """Sentence, span or document pairs to score (order matters for some scoring
    methods such as rouge).  Depending on the scoring method the ordering of the
    sentence pairs should be:

      * ``(<summary>, <source>)``

      * ``(<gold>, <prediction>)``

      * ``(<references>, <candidates>)``

    See :class:`.ScoreMethod` implementations for more information about pair
    ordering.

    """
    methods: Set[str] = field(default=None)
    """A set of strings, each indicating the :class:`.ScoreMethod` used to score
    :obj:`pairs`.

    """
    norm: bool = field(default=True)
    """Whether to use the normalized tokens, otherwise use the original text."""

    correlation_ids: Tuple[Union[int, str]] = field(default=None)
    """The IDs to correlate with each sentence pair, or ``None`` to skip
    correlating them.  The length of this tuple must be that of :obj:`pairs`.

    """
    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.correlation_ids is not None and \
           len(self.pairs) != len(self.correlation_ids):
            raise ScorerError(
                'Expecting same length pairs to correlation IDs but got: ' +
                f'{len(self.pairs)} != {len(self.correlation_ids)}')


@dataclass
class ScoreMethod(metaclass=ABCMeta):
    """An abstract base class for scoring methods (bleu, rouge, etc).

    """
    reverse_sents: bool = field(default=False)
    """Whether to reverse the order of the sentences."""

    @classmethod
    def _get_external_modules(cls: Type) -> Tuple[PackageRequirement, ...]:
        """Return a list of external module names needed by this method."""
        return ()

    @classmethod
    def missing_modules(cls: Type) -> Tuple[str, ...]:
        """Return a list of missing modules neede by this score method."""
        missing: List[str] = []
        mod: PackageRequirement
        for mod in cls._get_external_modules():
            try:
                ClassImporter.get_module(mod.name)
            except ModuleNotFoundError:
                missing.append(mod)
        return missing

    @classmethod
    def is_available(cls: Type) -> bool:
        """Whether or not this method is available on this system."""
        return len(cls.missing_modules()) == 0

    @abstractmethod
    def _score(self, meth: str, context: ScoreContext) -> Iterable[Score]:
        """See :meth:`score`"""
        pass

    def score(self, meth: str, context: ScoreContext) -> Iterable[Score]:
        """Score the sentences in ``context`` using method identifer ``meth``.

        :param meth: the identifer such as ``bleu``

        :param context: the context containing the data to score

        :return: the results, which are usually :class:`float` or
                 :class:`.Score`

        """
        scores: Iterable[Score]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'scoring meth: {meth}, ' +
                         f'reverse: {self.reverse_sents}')
        if not isinstance(context.pairs[0][0], TokenContainer):
            raise ScorerError(f'Wrong type: {type(context.pairs[0][0])} ' +
                              f' for first item, expecting {TokenContainer}')
        if not isinstance(context.pairs[0][1], TokenContainer):
            raise ScorerError(f'Wrong type: {type(context.pairs[0][0])} ' +
                              f' for second item, expecting {TokenContainer}')
        try:
            if self.reverse_sents:
                prev_pairs = context.pairs
                try:
                    context.pairs = tuple(map(
                        lambda x: (x[1], x[0]), context.pairs))
                    scores = self._score(meth, context)
                finally:
                    context.pairs = prev_pairs
            else:
                scores = self._score(meth, context)
            # force generators to realize scores and force any raised exceptions
            scores = tuple(scores)
        except Exception as e:
            logger.info(e, exc_info=True)
            scores = tuple([ErrorScore(meth, e)] * len(context.pairs))
        return scores

    def _tokenize(self, context: ScoreContext) -> \
            Iterable[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        s1: TokenContainer
        s2: TokenContainer
        for s1, s2 in context.pairs:
            s1t: Tuple[str, ...]
            s2t: Tuple[str, ...]
            if context.norm:
                s1t = tuple(map(lambda t: t.norm, s1.token_iter()))
                s2t = tuple(map(lambda t: t.norm, s2.token_iter()))
            else:
                s1t = tuple(map(lambda t: t.text, s1.token_iter()))
                s2t = tuple(map(lambda t: t.text, s2.token_iter()))
            yield (s1t, s2t)


@dataclass
class ExactMatchScoreMethod(ScoreMethod):
    """A scoring method that return 1 for exact matches and 0 otherwise.

    """
    equality_measure: str = field(default='norm')
    """The method by which to compare, which is one of:

        * ``norm``: compare with :meth:`.TokenContainer.norm`

        * ``text``: compare with :obj:`.TokenContainer.text`

        * ``equal``: compare using a Python object ``__eq__`` equal compare,
                     which also compares the token values

    """
    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        s1t: TokenContainer
        s2t: TokenContainer
        for s1t, s2t in context.pairs:
            val: float
            if self.equality_measure == 'norm':
                val = 1. if s1t.norm == s2t.norm else 0.
            elif self.equality_measure == 'text':
                val = 1. if s1t.text == s2t.text else 0.
            elif self.equality_measure == 'equal':
                val = 1. if s1t == s2t else 0.
            else:
                raise ScorerError(
                    f"No equality measure: '{self.equality_measure}'")
            yield FloatScore(val)


@dataclass
class LevenshteinDistanceScoreMethod(ScoreMethod):
    """A scoring method that computes the Levenshtein distance.

    """
    form: str = field(default='canon')
    """The form of the of the text used for the evaluation, which is one of:

        * ``text``: the original text with :obj:`.TokenContainer.text`

        * ``norm``: the normalized text using :meth:`.TokenContainer.norm`

        * ``canon``: :obj:`.TokenContainer.canonical` to normalize out
          whitespace for better comparisons

    """
    normalize: bool = field(default=True)
    """Whether to normalize  the return value as the *distince  / the max length
    of both sentences*.

    """
    @classmethod
    def _get_external_modules(cls: Type) -> Tuple[str, ...]:
        return (PackageRequirement.from_spec('editdistance~=0.8.1'),)

    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        import editdistance

        def container_to_str(container: TokenContainer) -> str:
            return container.norm if self.use_norm else container.text

        s1t: TokenContainer
        s2t: TokenContainer
        for s1t, s2t in context.pairs:
            t1: str
            t2: str
            if self.form == 'text':
                # use the normalized canonical form
                t1, t2 = s1t.text, s2t.text
            elif self.form == 'norm':
                # use the normalized canonical form
                t1, t2 = s1t.norm, s2t.norm
            elif self.form == 'canon':
                # use the normalized canonical form
                t1, t2 = s1t.canonical, s2t.canonical
            else:
                raise ScorerError(f"No form: '{self.form}'")
            val: int = editdistance.eval(t1, t2)
            if self.normalize:
                text_len: int = max(len(t1), len(t2))
                val = 1. - (val / text_len)
            val: float = val
            yield FloatScore(val)


@dataclass
class BleuScoreMethod(ScoreMethod):
    """The BLEU scoring method using the :mod:`nltk` package.  The first
    sentences are the references and the second are the hypothesis.

    """
    smoothing_function: bleu.SmoothingFunction = field(default=None)
    """This is an implementation of the smoothing techniques for segment-level
    BLEU scores.

    Citation:

      `Chen and Cherry (2014)`_ A Systematic Comparison of Smoothing Techniques
      for Sentence-Level BLEU. In WMT14.

    .. _Chen and Cherry (2014): http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf

    """
    weights: Tuple[float, ...] = field(default=(0.25, 0.25, 0.25, 0.25))
    """Weights for each n-gram.  For example: a tuple of float weights for
    unigrams, bigrams, trigrams and so on can be given: ``weights = (0.1, 0.3,
    0.5, 0.1)``.

    """
    silence_warnings: bool = field(default=False)
    """Silence the BLEU warning of n-grams not matching ``The hypothesis
    contains 0 counts of 3-gram overlaps...``

    """
    def __post_init__(self):
        if self.silence_warnings:
            import warnings
            # silence the BLEU warning of n-grams not matching
            # The hypothesis contains 0 counts of 3-gram overlaps...
            warnings.filterwarnings(
                'ignore', message='[.\n]+The hypothesis contains 0 counts.*')

    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        s1t: TokenContainer
        s2t: TokenContainer
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
    @classmethod
    def _get_external_modules(cls: Type) -> Tuple[str, ...]:
        return (PackageRequirement.from_spec('rouge_score~=0.1.2'),)

    def _score(self, meth: str, context: ScoreContext) -> \
            Iterable[HarmonicMeanScore]:
        from rouge_score import rouge_scorer

        class Tokenizer(object):
            @staticmethod
            def tokenize(sent: TokenContainer) -> Tuple[str, ...]:
                return sents[id(sent)]

        s1: TokenContainer
        s2: TokenContainer
        if self.feature_tokenizer:
            scorer = rouge_scorer.RougeScorer([meth], tokenizer=Tokenizer)
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
    (:obj:`methods`).

    """
    package_manager: PackageManager = field(default=None)
    """The package manager used to install scoring methods.  If this is
    ``None``, then packages are not installed and scoring methods are not made
    available.

    """
    methods: Dict[str, ScoreMethod] = field(default=None)
    """The registered scoring methods availale, which are accessed from
    :obj:`.ScoreContext.meth`.

    """
    default_methods: Set[str] = field(default=None)
    """Methods (keys from :obj:`methods`) to use when none are provided in the
    :obj:`.ScoreContext.meth` in the call to :meth:`score`.

    """
    def _install_all(self, reqs: Tuple[PackageRequirement, ...]) -> bool:
        if self.package_manager is None:
            return False
        else:
            req: PackageRequirement
            for req in reqs:
                try:
                    output: str = self.package_manager.install(req)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f'installed: {req}')
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'install output: <<{output}>>')
                except PackageError as e:
                    logger.warning(
                        f"could not install scoring requirement '{req}': {e}")
                    return False
            return True

    @persisted('_get_missing_modules_pw', cache_global=True)
    def _get_missing_modules(self) -> Tuple[PackageRequirement, ...]:
        missing: List[PackageRequirement] = []
        not_avail: List[str] = []
        name: str
        meth: ScoreMethod
        for name, meth in self.methods.items():
            missing: Tuple[PackageRequirement, ...] = meth.missing_modules()
            if len(missing) > 0 and not self._install_all(missing):
                logger.warning(f'method {meth} is not available: ' +
                               f'missing {missing}')
                not_avail.append(name)
                missing.extend(missing)
        for name in not_avail:
            del self.methods[name]
        return tuple(missing)

    def score(self, context: ScoreContext) -> ScoreSet:
        """Score the sentences in ``context``.

        :param context: the context containing the data to score

        :return: the results for each method indicated in ``context``

        """
        by_meth: Dict[str, Tuple[Score, ...]] = {}
        by_res: List[ScoreResult] = []
        meths: Iterable[str] = context.methods
        if meths is None:
            if self.default_methods is None:
                meths = self.methods.keys()
            else:
                meths = self.default_methods
        self._get_missing_modules()
        meth: str
        for meth in meths:
            smeth: ScoreMethod = self.methods.get(meth)
            if smeth is None:
                raise ScorerError(f"No scoring method: '{meth}'")
            by_meth[meth] = tuple(smeth.score(meth, context))
        for i in range(len(context.pairs)):
            item_res: Dict[str, Score] = {}
            corr_id: str = None
            meth: str
            if context.correlation_ids is not None:
                corr_id = context.correlation_ids[i]
            res_tup: Tuple[Score, ...]
            # for each scored pair
            for meth, res_tup in by_meth.items():
                item_res[meth] = res_tup[i]
            by_res.append(ScoreResult(item_res, correlation_id=corr_id))
        return ScoreSet(results=tuple(by_res))

    def __call__(self, context: ScoreContext) -> ScoreSet:
        """See :meth:`score`."""
        return self.score(context)
