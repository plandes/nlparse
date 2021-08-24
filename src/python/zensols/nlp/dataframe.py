"""Create Pandas dataframes from features.  This must be imported by absolute
module (:mod:`zensols.nlp.dataframe`).

"""
__author__ = 'Paul Landes'


from typing import Set, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
from zensols.nlp import TokenFeatures, FeatureDocument


@dataclass
class FeatureDataFrameFactory(object):
    """Creates a Pandas dataframe of features from a document annotations.

    """
    token_feature_ids: Set[str] = field(default=TokenFeatures.FIELD_IDS)
    priority_feature_ids: Tuple[str] = field(
        default=TokenFeatures.WRITABLE_FIELD_IDS)

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
