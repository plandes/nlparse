#!/usr/bin/env python

"""Example of how to score sentences using the :mod:`zensols.nlp.score` module.

"""
__author__ = 'Paul Landes'


from io import StringIO
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureSentenceFactory
from zensols.nlp.score import (
    Scorer, ScoreContext, ScoreSet, ScoreResult, HarmonicMeanScore
)

# application config
CONFIG = """
[import]
sections = list: imp_score

# we only need the score library since we're using FeatureSentenceFactory
[imp_score]
type = import
config_file = resource(zensols.nlp): resources/score.yml

# only use rouge1; comment out to get them all
[nlp_scorer]
methods: instance: dict: {'rouge1': 'nlp_rouge_score_method'}
"""


def main():
    # create the application config and factory
    fac = ImportConfigFactory(ImportIniConfig(StringIO(CONFIG)))
    # creates FeatureDocuments by splitting on whitespace
    sfac = FeatureSentenceFactory()
    # get the scorer from the application config
    scorer: Scorer = fac('nlp_scorer')
    # create the score context
    ctx = ScoreContext(pairs=(
        (
            sfac('I love cheese .'),
            sfac('I really love cheese .')
        ),
        (
            sfac('cheese I love .'),
            sfac('I really love cheese .')
        )))
    # score the sentences
    ss: ScoreSet = scorer(ctx)
    # get the first result and print out the R1 score
    sr: ScoreResult = ss.results[0]
    score: HarmonicMeanScore = sr['rouge1']
    score.write()


if (__name__ == '__main__'):
    main()
