# cython: language_level=3

VERSION = (0, 5, 7)
AUTHOR = "Michael Axiak"

__VERSION__ = VERSION

cimport cbloomfilter
cimport cpython

import array
import math
import random

cdef extern int errno
cdef NoConstruct = object()

cdef class BloomFilter:
    """
    Creates a new BloomFilter object with a given capacity and error_rate.

    :param int capacity: the maximum number of elements this filter
        can contain while keeping the false positive rate under ``error_rate``.
    :param float error_rate: false positive probability that will hold
        given that ``capacity`` is not exceeded.
    :param list hash_seeds: optionally specify hash seeds to use for the
        hashing algorithm. Each hash seed must not exceed 32 bits. The number
        of hash seeds will determine the number of hashes performed.
    :param bytes data_array: optionally specify the filter data array, same as
        given by BloomFilter.data_array. Only valid for in-memory bloomfilters.
        If provided, hash_seeds must be given too.

    **Note that we do not check capacity.** This is important, because
    we want to be able to support logical OR and AND (see :meth:`BloomFilter.union`
    and :meth:`BloomFilter.intersection`). The capacity and error_rate then together
    serve as a contract -- you add less than capacity items, and the Bloom filter
    will have an error rate less than error_rate.

    Raises :class:`OSError` if the supplied filename does not exist or if user
    lacks permission to access such file. Raises :class:`ValueError` if the supplied
    ``error_rate`` is invalid, ``hash_seeds`` does not contain valid hash seeds, or
    if the file cannot be read.
    """

    cdef cbloomfilter.BloomFilter * _bf

    def __reduce__(self):
        """Makes an in-memory BloomFilter pickleable."""
        callable = BloomFilter
        args = (self.capacity, self.error_rate, self.hash_seeds, self.data_array)
        return (callable, args)

    def __cinit__(self, capacity, error_rate, hash_seeds=None, data_array=None):
        if capacity is NoConstruct:
            return

        self._create(capacity, error_rate, hash_seeds, data_array)

    def _create(self, capacity, error_rate, hash_seeds=None, data_array=None):
        cdef char * seeds
        cdef char * data = NULL
        cdef long long num_bits

        if data_array is not None:
            if hash_seeds is None:
                raise ValueError("hash_seeds must be specified if a data_array is provided.")

        # For why we round down for determining the number of hashes:
        # http://corte.si/%2Fposts/code/bloom-filter-rules-of-thumb/index.html
        # "The number of hashes determines the number of bits that need to
        # be read to test for membership, the number of bits that need to be
        # written to add an element, and the amount of computation needed to
        # calculate hashes themselves. We may sometimes choose to use a less
        # than optimal number of hashes for performance reasons (especially
        # when we choose to round down when the calculated optimal number of
        # hashes is fractional)."

        if not (0 < error_rate < 1):
            raise ValueError("error_rate allowable range (0.0, 1.0) %f" % (error_rate,))

        array_seeds = array.array('I')

        if hash_seeds:
            for seed in hash_seeds:
                if not isinstance(seed, int) or seed < 0 or seed.bit_length() > 32:
                    raise ValueError("invalid hash seed '%s', must be >= 0 "
                                     "and up to 32 bits in size" % seed)
            num_hashes = len(hash_seeds)
            array_seeds.extend(hash_seeds)
        else:
            num_hashes = max(math.floor(math.log2(1 / error_rate)), 1)
            array_seeds.extend([random.getrandbits(32) for i in range(num_hashes)])

        test = array_seeds.tobytes()
        seeds = test

        bits_per_hash = math.ceil(
            capacity * abs(math.log(error_rate)) /
            (num_hashes * (math.log(2) ** 2)))

        # Minimum bit vector of 128 bits
        num_bits = max(num_hashes * bits_per_hash, 128)

        # Override calculated capacity if we are provided a data array
        if data_array is not None:
            num_bits = 8 * len(data_array)

        # print("k = %d  m = %d  n = %d   p ~= %.8f" % (
        #     num_hashes, num_bits, capacity,
        #     (1.0 - math.exp(- float(num_hashes) * float(capacity) / num_bits))
        #     ** num_hashes))

        if data_array is not None:
            data = data_array
        self._bf = cbloomfilter.bloomfilter_Create_Malloc(capacity,
                                                          error_rate,
                                                          num_bits,
                                                          <int *> seeds,
                                                          num_hashes, <const char *> data)
        if self._bf is NULL:
            cpython.PyErr_NoMemory()

    def __dealloc__(self):
        cbloomfilter.bloomfilter_Destroy(self._bf)
        self._bf = NULL

    @property
    def bit_array(self):
        """Bit vector representation of the Bloom filter contents.
        Returns an integer.
        """
        start_pos = self._bf.array.preamblebytes
        end_pos = start_pos + self._bf.array.bytes
        arr = (<char *> cbloomfilter.mbarray_CharData(self._bf.array))[start_pos:end_pos]
        return int.from_bytes(arr, byteorder="big", signed=False)

    @property
    def data_array(self):
        """Bytes array of the Bloom filter contents.
        """
        start_pos = self._bf.array.preamblebytes
        end_pos = start_pos + self._bf.array.bytes
        arr = array.array('B')
        arr.frombytes(
            (<char *> cbloomfilter.mbarray_CharData(self._bf.array))[start_pos:end_pos]
        )
        return bytes(arr)

    @property
    def hash_seeds(self):
        """Integer seeds used for the random hashing. Returns a list of integers."""
        seeds = array.array('I')
        seeds.frombytes(
            (<char *> self._bf.hash_seeds)[:4 * self.num_hashes]
        )
        return seeds

    @property
    def capacity(self):
        """The maximum number of elements this filter can contain while keeping
        the false positive rate under :attr:`BloomFilter.error_rate`.
        Returns an integer.
        """
        return self._bf.max_num_elem

    @property
    def error_rate(self):
        """The acceptable probability of false positives. Returns a float."""
        return self._bf.error_rate

    @property
    def num_hashes(self):
        """Number of hash functions used when computing."""
        return self._bf.num_hashes

    @property
    def num_bits(self):
        """Number of bits used in the filter as buckets."""
        return self._bf.array.bits

    @property
    def bit_count(self):
        """Number of bits set to one."""
        return cbloomfilter.mbarray_BitCount(self._bf.array)

    @property
    def approx_len(self):
        """Approximate number of items in the set.

        See also:
        - https://en.wikipedia.org/wiki/Bloom_filter#The_union_and_intersection_of_sets
        - https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1063.3591&rep=rep1&type=pdf
        """
        m = self.num_bits
        k = self.num_hashes
        X = self.bit_count

        n = -(m / k) * math.log(1 - (X / m), math.e)
        return round(n)

    def __repr__(self):
        my_name = self.__class__.__name__
        return '<%s capacity: %d, error: %0.3f, num_hashes: %d>' % (
            my_name, self._bf.max_num_elem, self._bf.error_rate,
            self._bf.num_hashes)

    def __str__(self):
        return self.__repr__()

    def clear_all(self):
        """Removes all elements from the Bloom filter at once."""
        cbloomfilter.bloomfilter_Clear(self._bf)

    def __contains__(self, item_):
        """Checks to see if item is contained in the filter, with
        an acceptable false positive rate of :attr:`BloomFilter.error_rate`.

        :param item: hashable object
        :rtype: bool
        """
        cdef cbloomfilter.Key key
        if isinstance(item_, str):
            item = item_.encode()
            key.shash = item
            key.nhash = len(item)
        elif isinstance(item_, bytes):
            item = item_
            key.shash = item
            key.nhash = len(item)
        else:
            # Warning! Only works reliably for objects whose hash is based on value not memory address.
            item = item_
            key.shash = NULL
            key.nhash = hash(item)
        return cbloomfilter.bloomfilter_Test(self._bf, &key) == 1

    def add(self, item_):
        """Adds an item to the Bloom filter. Returns a boolean indicating whether
        this item was present in the Bloom filter prior to adding
        (see :meth:`BloomFilter.__contains__`).

        :param item: hashable object
        :rtype: bool
        """
        cdef cbloomfilter.Key key
        if isinstance(item_, str):
            item = item_.encode()
            key.shash = item
            key.nhash = len(item)
        elif isinstance(item_, bytes):
            item = item_
            key.shash = item
            key.nhash = len(item)
        else:
            item = item_
            key.shash = NULL
            key.nhash = hash(item)

        result = cbloomfilter.bloomfilter_Add(self._bf, &key)
        if result == 2:
            raise RuntimeError("Some problem occured while trying to add key.")
        return bool(result)

    def update(self, iterable):
        """Calls :meth:`BloomFilter.add` on all items in the iterable."""
        for item in iterable:
            self.add(item)

    def __len__(self):
        """Returns the number of distinct elements that have been
        added to the :class:`BloomFilter` object, subject to the error
        given in :attr:`BloomFilter.error_rate`.

        The length reported here is exact as long as no set `union` or
        `intersection` were performed. Otherwise we report an approximation
        of based on :attr:`BloomFilter.bit_count`.

        :param item: hashable object
        :rtype: int
        """
        if not self._bf.count_correct:
            return self.approx_len
        return self._bf.elem_count

    def union(self, BloomFilter other):
        """Performs a set OR with another comparable filter. You can (only) construct
        comparable filters with :meth:`BloomFilter.copy_template` above.

        The computation will occur *in place*. That is, calling::

            >>> bf.union(bf2)

        is a way of adding *all* the elements of ``bf2`` to ``bf``.

        *NB: Calling this function will render future calls to len()
        invalid.*

        :param BloomFilter other: filter to perform the union with
        :rtype: :class:`BloomFilter`
        """
        self._assert_comparable(other)
        cbloomfilter.mbarray_Or(self._bf.array, other._bf.array)
        self._bf.count_correct = 0
        return self

    def __ior__(self, BloomFilter other):
        """See :meth:`BloomFilter.union`."""
        return self.union(other)

    def intersection(self, BloomFilter other):
        """The same as :meth:`BloomFilter.union` above except it uses
        a set AND instead of a set OR.

        *NB: Calling this function will render future calls to len()
        invalid.*

        :param BloomFilter other: filter to perform the intersection with
        :rtype: :class:`BloomFilter`
        """
        self._assert_comparable(other)
        cbloomfilter.mbarray_And(self._bf.array, other._bf.array)
        self._bf.count_correct = 0
        return self

    def __iand__(self, BloomFilter other):
        """See :meth:`BloomFilter.intersection`."""
        return self.intersection(other)

    def _assert_comparable(self, BloomFilter other):
        error = ValueError("The two %s objects are not the same type (hint: "
                           "use copy_template)" % self.__class__.__name__)
        if self._bf.array.bits != other._bf.array.bits:
            raise error
        if self.hash_seeds != other.hash_seeds:
            raise error
        return
