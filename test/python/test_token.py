from unittest import TestCase
from io import StringIO
from zensols.nlp import FeatureToken


class TestToken(TestCase):
    def test_access(self):
        t = FeatureToken(1, 2, 3, 'cat')
        self.assertEqual(1, t.i)
        self.assertEqual('cat', t.norm)
        self.assertEqual({'i': 1, 'idx': 2, 'i_sent': 3, 'norm': 'cat'},
                         t.asdict())

        self.assertEqual((1, 3, 2, 'cat'), tuple(t.to_vector()))
        self.assertEqual(t, FeatureToken(1, 2, 3, 'cat'))
        self.assertNotEqual(t, FeatureToken(10, 2, 3, 'cat'))
        self.assertNotEqual(t, FeatureToken(1, 10, 3, 'cat'))
        self.assertNotEqual(t, FeatureToken(1, 2, 10, 'cat'))
        self.assertNotEqual(t, FeatureToken(1, 2, 3, 'dog'))

    def test_detach(self):
        t = FeatureToken(1, 2, 3, 'cat')
        det = t.detach(skip_missing=True)
        self.assertNotEqual(id(t), id(det))
        self.assertEqual(t, det)

    def test_write(self):
        t = FeatureToken(1, 2, 3, 'cat')
        sio = StringIO()
        t.write(writer=sio)
        self.assertEqual("""\
FeatureToken: norm=<cat>
    attributes:
        i=1 (int)
        i_sent=3 (int)
        idx=2 (int)
        norm=cat (str)\n""", sio.getvalue())
