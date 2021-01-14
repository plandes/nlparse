# Token Normalizers and Mappers List

This package provides a simple, yet robust way to generate a string stream of
tokens using a [TokenNormalizer] as mentioned in the [parsing](parse.md)
documentation (please read this first).

A full list of token normalizers mappers are listed below.  Note that the API
was written to easily extend to create your own using the [configuration
factory] API.

* [TokenNormalizer]: Base token extractor returns tuples of tokens and their
  normalized version.
* [TokenMapper]: Abstract class used to transform token tuples generated from
  `TokenNormalizer.normalize`.
* [MapTokenNormalizer]: A normalizer that applies a sequence of
  `TokenMapper`s to transform the normalized token text.
* [SplitTokenMapper]: Splits the normalized text on a per token basis with a
  regular expression.
* [LemmatizeTokenMapper]: Lemmatize tokens and optional remove entity stop
  words.
* [FilterTokenMapper]: Filter tokens based on token (Spacy) attributes.
* [SubstituteTokenMapper]: Replace a string in normalized token text.
* [LambdaTokenMapper]: Use a lambda expression to map a token tuple.


<!-- links -->
[configuration factory]: https://plandes.github.io/util/doc/config.html#configuration-factory

[TokenNormalizer]: ../api/zensols.nlp.html#zensols.nlp.norm.TokenNormalizer
[TokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.TokenMapper
[MapTokenNormalizer]: ../api/zensols.nlp.html#zensols.nlp.norm.MapTokenNormalizer
[SplitTokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.SplitTokenMapper
[LemmatizeTokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.LemmatizeTokenMapper
[FilterTokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.FilterTokenMapper
[SubstituteTokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.SubstituteTokenMapper
[LambdaTokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.LambdaTokenMapper
