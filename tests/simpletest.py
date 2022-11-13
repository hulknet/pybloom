import os
import string
import unittest
import tempfile
from random import randint, choice, getrandbits

import pybloomfilter

from tests import with_test_file


class SimpleTestCase(unittest.TestCase):
    FILTER_SIZE = 200
    FILTER_ERROR_RATE = 0.001

    def setUp(self):
        # Convenience memory-backed bloomfilter
        self.bf_mem = pybloomfilter.BloomFilter(self.FILTER_SIZE,
                                                self.FILTER_ERROR_RATE)

    def assertPropertiesPreserved(self, old_bf, new_bf):
        # Assert that a "new" BloomFilter has the same properties as an "old"
        # one.
        failures = []
        for prop in ['capacity', 'error_rate', 'num_hashes', 'num_bits',
                     'hash_seeds']:
            old, new = getattr(old_bf, prop), getattr(new_bf, prop)
            if new != old:
                failures.append((prop, old, new))
        self.assertEqual([], failures)

    def _random_str(self, length=16):
        chars = string.ascii_letters
        return ''.join(choice(chars) for _ in range(length))

    def _random_set_of_stuff(self, c):
        """
        Return a random set containing up to "c" count of each type of Python
        object.
        """
        return set(
            # Due to a small chance of collision, there's no guarantee on the
            # count of elements in this set, but we'll make sure that's okay.
            [self._random_str() for _ in range(c)] +
            [randint(-1000, 1000) for _ in range(c)] +
            [(randint(-200, 200), self._random_str()) for _ in range(c)] +
            [float(randint(10, 100)) / randint(10, 100)
             for _ in range(c)] +
            [int(randint(50000, 1000000)) for _ in range(c)] +
            [object() for _ in range(c)] +
            [str(self._random_str) for _ in range(c)])

    def _populate_filter(self, bf, use_update=False):
        """
        Populate given BloomFilter with a handfull of hashable things.
        """
        self._in_filter = self._random_set_of_stuff(10)
        self._not_in_filter = self._random_set_of_stuff(15)
        # Just in case we randomly chose a key which was also in
        # self._in_filter...
        self._not_in_filter = self._not_in_filter - self._in_filter

        if use_update:
            bf.update(self._in_filter)
        else:
            for item in self._in_filter:
                bf.add(item)

    def _check_filter_contents(self, bf):
        for item in self._in_filter:
            # We should *never* say "not in" for something which was added
            self.assertTrue(item in bf, '%r was NOT in %r' % (item, bf))

        # We might say something is in the filter which isn't; we're only
        # trying to test correctness, here, so we are very lenient.  If the
        # false positive rate is within 2 orders of magnitude, we're okay.
        false_pos = len(list(filter(bf.__contains__, self._not_in_filter)))
        error_rate = float(false_pos) / len(self._not_in_filter)
        self.assertTrue(error_rate < 100 * self.FILTER_ERROR_RATE,
                        '%r / %r = %r > %r' % (false_pos,
                                               len(self._not_in_filter),
                                               error_rate,
                                               100 * self.FILTER_ERROR_RATE))
        for item in self._not_in_filter:
            # We should *never* have a false negative
            self.assertFalse(item in bf, '%r WAS in %r' % (item, bf))

    def test_repr(self):
        self.assertEqual(
            '<BloomFilter capacity: %d, error: %0.3f, num_hashes: %d>' % (
                self.bf_mem.capacity, self.bf_mem.error_rate, self.bf_mem.num_hashes),
            repr(self.bf_mem))
        self.assertEqual(
            '<BloomFilter capacity: %d, error: %0.3f, num_hashes: %d>' % (
                self.bf_mem.capacity, self.bf_mem.error_rate, self.bf_mem.num_hashes),
            str(self.bf_mem))
        self.assertEqual(
            '<BloomFilter capacity: %d, error: %0.3f, num_hashes: %d>' % (
                self.bf_mem.capacity, self.bf_mem.error_rate, self.bf_mem.num_hashes),
            str(self.bf_mem))

    def test_add_and_check_memory_backed(self):
        self._populate_filter(self.bf_mem)
        self._check_filter_contents(self.bf_mem)

    def test_number_nofile(self):
        bf = pybloomfilter.BloomFilter(100, 0.01)
        bf.add(1234)
        self.assertEqual(1234 in bf, True)

    def test_string_nofile(self):
        bf = pybloomfilter.BloomFilter(100, 0.01)
        bf.add("test")
        self.assertEqual("test" in bf, True)

    def test_others_nofile(self):
        bf = pybloomfilter.BloomFilter(100, 0.01)
        for elem in (1.2, 2343, (1, 2), object(), '\u2131\u3184'):
            bf.add(elem)
            self.assertEqual(elem in bf, True)

    def test_create_with_hash_seeds_invalid(self):
        cust_seeds = ["ABC", -123, "123456", getrandbits(33)]
        self.assertRaises(ValueError,
                          pybloomfilter.BloomFilter,
                          self.FILTER_SIZE,
                          self.FILTER_ERROR_RATE,
                          hash_seeds=cust_seeds)

    def test_create_with_hash_seeds_and_compare(self):
        test_data = "test"
        bf1 = pybloomfilter.BloomFilter(self.FILTER_SIZE,
                                        self.FILTER_ERROR_RATE)
        bf1.add(test_data)
        bf1_seeds = bf1.hash_seeds.tolist()
        bf1_ba = bf1.bit_array

        bf2 = pybloomfilter.BloomFilter(self.FILTER_SIZE,
                                        self.FILTER_ERROR_RATE,
                                        hash_seeds=bf1_seeds)
        bf2.add(test_data)
        bf2_seeds = bf2.hash_seeds.tolist()
        bf2_ba = bf2.bit_array

        self.assertEqual(bf1_seeds, bf2_seeds)

        # Expecting same hashing sequence
        self.assertEqual(bf1_ba, bf2_ba)

    def test_bit_array(self):
        bf = pybloomfilter.BloomFilter(1000, 0.01)
        bf.add("apple")

        # Count the number of 1s
        total_ones = 0
        bit_array_str = bin(bf.bit_array)
        for c in bit_array_str:
            if c == "1":
                total_ones += 1

        # For the first item addition, BF should contain
        # the same amount of 1s as the number of hashes
        # performed
        assert total_ones == bf.num_hashes

        for i in range(1000):
            bf.add(randint(0, 1000))

        bf.add("apple")
        ba_1 = bf.bit_array

        bf.add("apple")
        ba_2 = bf.bit_array

        # Should be the same
        assert ba_1 ^ ba_2 == 0

        bf.add("pear")
        bf.add("mango")
        ba_3 = bf.bit_array

        # Should not be the same
        assert ba_1 ^ ba_3 != 0

    def test_bit_array_same_hashes(self):
        capacity = 100 * 100
        items = []
        for i in range(capacity):
            items.append(randint(0, 1000))

        # File-backed
        bf1 = pybloomfilter.BloomFilter(capacity, 0.01)
        bf1.update(items)

        bf1_hs = bf1.hash_seeds
        bf1_ba = bf1.bit_array

        # In-memory
        bf2 = pybloomfilter.BloomFilter(capacity, 0.01, hash_seeds=bf1_hs)
        bf2.update(items)

        bf2_ba = bf2.bit_array

        # Should be identical as data was hashed into the same locations
        assert bf1_ba ^ bf2_ba == 0

    def test_add_contains(self):
        values = [
            '',
            'string',
            b'',
            b'bytes',
            True,
            False,
            0,
            10,
        ]
        for value in values:
            bf = pybloomfilter.BloomFilter(1000, 0.1)
            bf.add(value)
            assert value in bf

    def test_bit_count(self):
        bf0 = pybloomfilter.BloomFilter(100, 0.1)
        bf1 = pybloomfilter.BloomFilter(100, 0.1)
        bf1.add('a')
        bf100 = pybloomfilter.BloomFilter(100, 0.1)
        for i in range(100):
            bf100.add(str(i))

        assert bf0.bit_count == 0
        assert bf1.bit_count == bf1.num_hashes
        assert bf100.bit_count == bin(bf100.bit_array).count('1')

    def test_approximate_size_after_union_called(self):
        bf1 = pybloomfilter.BloomFilter(100, 0.1, hash_seeds=[1, 2, 3])
        for i in range(0, 20):
            bf1.add(str(i))
        bf_union = pybloomfilter.BloomFilter(100, 0.1, hash_seeds=[1, 2, 3])
        for i in range(0, 20):
            bf_union.add(str(i))

        bf2 = pybloomfilter.BloomFilter(100, 0.1, hash_seeds=[1, 2, 3])
        for i in range(10, 30):  # intersectoin size: 10
            bf2.add(str(i))

        bf_union.union(bf2)

        assert len(bf_union) == 29  # approximate size
        intersection = len(bf1) + len(bf2) - len(bf_union)
        assert intersection == 11  # approximate size


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SimpleTestCase))
    return suite
