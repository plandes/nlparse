from unittest import TestCase
from zensols.nlp import LexicalSpan


class TestOverlap(TestCase):
    def test_attr(self):
        loc = LexicalSpan(2, 5)
        self.assertEqual(loc[0], 2)
        self.assertEqual(loc[1], 5)
        with self.assertRaises(KeyError):
            loc[2]
        self.assertEqual(3, len(loc))
        self.assertEqual('[2, 5]', str(loc))
        self.assertTrue(isinstance(hash(loc), int))

    def test_order(self):
        a = LexicalSpan(2, 5)
        b = LexicalSpan(2, 5)
        c = LexicalSpan(3, 6)
        d = LexicalSpan(6, 10)
        self.assertNotEqual(id(a), id(b))
        self.assertEqual(a, b)
        self.assertLess(a, c)
        self.assertGreater(c, a)
        self.assertLess(a, d)
        self.assertGreater(d, a)
        self.assertLess(c, d)
        self.assertGreater(d, c)

    def test_overlap_meth(self):
        # case 1: both overlap
        self.assertTrue(LexicalSpan.overlaps(51, 1253, 51, 1253))

        # case 2: first proceeds: beg overllap
        self.assertTrue(LexicalSpan.overlaps(51, 1253, 51, 2565))
        self.assertTrue(LexicalSpan.overlaps(51, 1253, 52, 2565))
        self.assertTrue(LexicalSpan.overlaps(51, 1253, 1253, 2565))

        # case 3: first proceeds: disjoint
        self.assertFalse(LexicalSpan.overlaps(51, 1253, 1254, 2565))
        self.assertFalse(LexicalSpan.overlaps(51, 1253, 1255, 2565))

        # case 4: second proceeds: beg overllap
        self.assertTrue(LexicalSpan.overlaps(51, 2565, 51, 1253))
        self.assertTrue(LexicalSpan.overlaps(52, 2565, 51, 1253))
        self.assertTrue(LexicalSpan.overlaps(1253, 2565, 51, 1253))

        # case 5: second proceeds: disjoint
        self.assertFalse(LexicalSpan.overlaps(1254, 2565, 51, 1253))
        self.assertFalse(LexicalSpan.overlaps(1255, 2565, 51, 1253))

    def test_overlap_inst(self):
        # case 1: both overlap
        self.assertTrue(LexicalSpan(51, 1253).overlaps_with(LexicalSpan(51, 1253)))

        # case 2: first proceeds: beg overllap
        self.assertTrue(LexicalSpan(51, 1253).overlaps_with(LexicalSpan(51, 2565)))
        self.assertTrue(LexicalSpan(51, 1253).overlaps_with(LexicalSpan(52, 2565)))
        self.assertTrue(LexicalSpan(51, 1253).overlaps_with(LexicalSpan(1253, 2565)))

        # case 3: first proceeds: disjoint
        self.assertFalse(LexicalSpan(51, 1253).overlaps_with(LexicalSpan(1254, 2565)))
        self.assertFalse(LexicalSpan(51, 1253).overlaps_with(LexicalSpan(1255, 2565)))

        # case 4: second proceeds: beg overllap
        self.assertTrue(LexicalSpan(51, 2565).overlaps_with(LexicalSpan(51, 1253)))
        self.assertTrue(LexicalSpan(52, 2565).overlaps_with(LexicalSpan(51, 1253)))
        self.assertTrue(LexicalSpan(1253, 2565).overlaps_with(LexicalSpan(51, 1253)))

        # case 5: second proceeds: disjoint
        self.assertFalse(LexicalSpan(1254, 2565).overlaps_with(LexicalSpan(51, 1253)))
        self.assertFalse(LexicalSpan(1255, 2565).overlaps_with(LexicalSpan(51, 1253)))
