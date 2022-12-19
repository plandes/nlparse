"""Create Pandas dataframes from features.  This must be imported by absolute
module (:mod:`zensols.nlp.dataframe`).

"""
__author__ = 'Paul Landes'


from typing import Set, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
from zensols.nlp import FeatureToken, FeatureDocument


@dataclass
class FeatureDataFrameFactory(object):
    """Creates a Pandas dataframe of features from a document annotations.  Each
    feature ID is given a column in the output :class:`pandas.DataFrame`.

    """
    token_feature_ids: Set[str] = field(default=FeatureToken.FEATURE_IDS)
    """The feature IDs to add to the :class:`pandas.DataFrame`."""

    priority_feature_ids: Tuple[str, ...] = field(
        default=FeatureToken.WRITABLE_FEATURE_IDS)
    """Feature IDs that are used first in the column order in the output
    :class:`pandas.DataFrame`.

    """
    def __call__(self, doc: FeatureDocument) -> pd.DataFrame:
        fids = self.token_feature_ids
        cols: List[str] = list(filter(lambda n: n in fids,
                                      self.priority_feature_ids))
        cols.extend(sorted(fids - set(cols)))
        rows = []
        for six, sent in enumerate(doc.sents):
            for tok in sent:
                feats = tok.asdict()
                rows.append(tuple(map(lambda f: feats.get(f), cols)))
        return pd.DataFrame(rows, columns=cols)
