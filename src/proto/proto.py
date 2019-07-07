import logging
import importlib
from zensols.nlp import AppConfig

logger = logging.getLogger(__name__)


def create_config():
    return AppConfig('resources/nlparse.conf')


def tmp():
    import zensols.nlp.app
    importlib.reload(zensols.nlp.app)
    logging.getLogger('zensols.actioncli').setLevel(logging.INFO)
    logging.getLogger('zensols.nlp.app').setLevel(logging.DEBUG)
    app = zensols.nlp.app.MainApplication(create_config())
    app.tmp()


def main():
    logging.basicConfig(level=logging.WARNING)
    run = 1
    {1: tmp,
     }[run]()


main()
