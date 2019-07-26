import logging

logger = logging.getLogger(__name__)


def test():
    from zensols.actioncli import ClassImporter
    import sys
    sys.path.append('./test/python')
    inst = ClassImporter('test_nlparse.TestParse').instance()
    #logging.getLogger('zensols.nlp').setLevel(level=logging.DEBUG)
    inst.setUp()
    inst.test_disable()
    inst.tearDown()


def main():
    logging.basicConfig(level=logging.WARNING)
    run = 1
    {1: test,
     }[run]()


main()
