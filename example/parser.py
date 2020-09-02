#!/usr/bin/env python

from zensols.config import ExtendedInterpolationEnvConfig, ImportConfigFactory


def create_factory():
    conf = ExtendedInterpolationEnvConfig('parser.conf')
    return ImportConfigFactory(conf)


def main():
    sent = 'California is part of the United States.'
    fac = create_factory()
    lr = fac('langres')
    doc = lr.parse(sent)
    print(type(doc))
    for tok in doc:
        print(tok, tok.tag_, tok.is_stop)
    print('-' * 10)
    feats = lr.features(doc)
    for feat in feats:
        print(f'{feat} {type(feat)}')
        feat.write(depth=1)
        print('-' * 5)
        det = feat.detach()
        print(f'detached: {type(det)}: {det.to_dict()}')
        print('-' * 5)
    print(', '.join(lr.normalized_tokens(doc)))
    print('-' * 10)

    lr = fac('lc_langres')
    doc = lr.parse(sent)
    print(', '.join(lr.normalized_tokens(doc)))


if __name__ == '__main__':
    main()


main()
