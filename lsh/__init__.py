from collections import defaultdict
import itertools as it
import random
import sys
import time
import logging
from math import sqrt

logging.getLogger().setLevel(logging.INFO)

class LSHCache:
    """
    Locality-Sensitive Hashing (LSH) implementation as described in 
    Mining of Massive Dataset, Chapter 3 by Anand Rajaraman and Jeff Ullman, Chapter 3
    http://infolab.stanford.edu/~ullman/mmds/ch3.pdf
    
    A document is a sequence of items which will be shingled together to look for
    similarity.
    
    This can be sub-classed to allow for implementing your own method for hashing shingles
    or hashing a shingle into the set of hashes used by minhash.
    
    Override _minhash_hash and _minhash_init in order to change how minhashes are calculated
    
    Override _hash_shingle in order to change how shingles are hashed. The default behavior
    is to simply hash each tuple.  Another approach would be to keep a cache mapping each shingle
    seen to a unique id and return that.
    """
    
    def __init__(self, b=None, r=None, n=None,
                 min_shingle=None, max_shingle=None, shingle_len=None,
                 store_signatures = False, dups_on_insert = True, seed=None):
        """
        An implementation of Locality-Sensitive Hashing (LSH) using minhash
        
        All arguments are optional. If no additional arguments are specified, 
        the LSH will be 20 bands of 5 rows each (100 total rows) with a shingle length of 2.
        
        You can specify the number of bands and rows or just the total number of rows
        and either bands and rows-per-band.  If the specification is incomplete, it will determine
        missing values based on the 2 out of 3 specified (given that b*r==n).  If only total number
        of rows is specified, it will factor that total number to the most even split
        between bands and rows-per-band.  The arguments to specify are
            b: number of bands
            r: number of rows per band
            n: total number of rows
        
        You may also specify the shingling of documents.  If no values are specified, it
        will a shingle length of 2.  You may specify using multiple shingle lengths by specifying
        min_shingle and max_shingle instead of shingle_len.  The arguments are:
            shingle_len: length of shingle to use (cannot be specified with min_shingle and max_shingle)
            min_shingle: minimum length of shingle to use (cannot be specified with shingle_len)
            max_shingle: maximum length of shingle to use (cannot be specified with shingle_len)
            
        Finally, there are a few additional optional arguments.
            store_signatures: whether to store the generated signatures.  This allows later lookups to
                              be done by doc_id rather than having to rehash a document in the cache. By default,
                              signatures are not stored
            dups_on_insert:   whether to return the duplicates found in the cache when a new document is
                              inserted.  If False, it returns the generated doc_id for the inserted document.
                              By default, True
            seed:             seed to set before calling random to set the random bits used for hashing
        """

        # default to 20 bands of 5 rows         
        if n is None and r is None and b is None:
            # defaults
            b,r = (20,5)

        if n is None:
            assert r is not None and b is not None, "Must specify number of rows and bands"
            n = b*r
        elif b is None and r is not None:
            assert n % r == 0, "total rows is not divisible by number of rows per band"
            b = n / r
        elif r is None and b is not None:
            assert n % b == 0, "total rows is not divisible by number of bands"
            r = n / b
        elif b is None and r is None:
            # calculate the most even distributaion of bands and rows given then total number of rows
            # There must be a much smarter way to do this, but hopefully the caller isn't spiteful
            for b in xrange(int(sqrt(n)), 1, -1):
                if n % b == 0:
                    r = n/b
                    break
            else:
                raise AssertionError("cannot reasonably divide a prime number of total rows (%d) into bands and rows per band" % n)
        assert b*r==n, "inconsistent specifications of rows and bands"
        
        # default shingle_len==2
        if shingle_len is None and min_shingle is None and max_shingle is None:
            shingle_len = 2
    
        if shingle_len is not None:
            assert min_shingle is None and max_shingle is None, "too many specifications of shingle length"
            min_shingle = max_shingle = shingle_len

        assert min_shingle > 0, 'shingle length must be greater than 0, not %d' % min_shingle
        if max_shingle is None:
            max_shingle = min_shingle
        else:
            assert max_shingle >= min_shingle, 'max_shingle must be greater than min_shingle'
        
        logging.debug("building LSH cache with %d total rows (%d bands, %d rows per band) with shingle length %s",
                      (n, b, r, 
                       ("<=".join(map(str,sorted(set((min_shingle,max_shingle,))))))
                       )
                      )
        # assign it
        self._b = b
        self._r = r
        self._n = n
        self._min_shingle = min_shingle
        self._max_shingle = max_shingle
        self._store_signatures = store_signatures
        self._Accumulator = self.AccumulatorDups if dups_on_insert else self.AccumulatorDocId

        # make it 
        self._seen = {} # the set of doc ids which have already been hashed
        self._next_id = 0
        self._cache = [defaultdict(list) for _ in xrange(self._b)]
        if seed is not None:
            random.seed(seed)
        self._init_minhash_hash()


    def _init_minhash_hash(self):
        """
        This initializes the instance variable _memomask which is a list of the 
        random 32 bits associated with each hash function
        """
        self._memomask = [ int(random.getrandbits(32)) for _ in xrange(self._n)]

    def _xor_hash(self,x, mask):
        """
        This is a simple hash function which returns the result of a bitwise XOR
        on the input x and the 32-bit random mask
        """
        return int(x ^ mask)

    def _minhash_hash(self, x):
        """ 
        generate the series of hashes of the value to be used for finding the minhash
        The implementation uses _xor_hashing with a series of random 32-bit fields
        """
        # trim x to 32-bits
        x = x & 0xffffffff
        return it.imap(lambda mask: self._xor_hash(x, mask), self._memomask)

    def _hash_shingle(self, shingle):
        return hash(shingle)
        
    def _get_shingle_vec(self, doc):
        """
        Takes a sequence of tokenized words and maps each shingle to a unique id.
        These unique ids, are then added to the shingle_vec object which is just a sparse
        vector implemented as a dict with v[id]=1 when a shingle id is present
        """
        logging.debug('entering with len(doc)=%d', len(doc))
        v = set()
        for n in xrange(self._min_shingle, self._max_shingle+1):
            if len(doc) < n:
                v.add(self._hash_shingle(('',)*(n-len(doc))+tuple(doc)))
            else:
                for j in xrange(len(doc) - (n-1)):
                    s = tuple(doc[j:j+n])
                    v.add(self._hash_shingle(s))
        return v

    def _get_sig(self,shingle_vec):
        """
        Takes a shingle vec and computes the minhash signature of length n using
        approximate permutations.  This method is explained in Mining Massive
        Datasets by Rajaraman and Ullman (http://infolab.stanford.edu/~ullman/mmds.html)
        in section 3.3.4.
        """
        mhash = [sys.maxint]*self._n
        for shingle in shingle_vec:
            #logging.debug('r=%d', r)
            for i,h in enumerate(self._minhash_hash(shingle)):
                if (h < mhash[i]):
                    mhash[i] = h
        return mhash
        
    def _get_lsh(self,sig):
        """
        Takes an n-dimensional minhash signature and computes b hashes for each of
        b bands of r rows in the signature.  These hashes can take on any value that
        can be stored in the 32bit integer.
        """
        lsh = [None]*self._b
        for i in xrange(self._b):
            lsh[i] = hash(tuple(sig[self._r*i:self._r*(i+1)]))
                                    
        #logging.debug('hashed signature: %s\n[get_lsh]\tto bins: %s', (sig,lsh)
        return lsh
    
    def _get_lsh_from_doc(self, doc):
        """
        given an iterable of hashable items, returns a list of bucket ids
        """
        logging.debug('got tokenized doc: len(doc)=%d', len(doc))
        shingle_vec = self._get_shingle_vec(doc)
        logging.debug('got shingle_vec: len(shingle_vec)=%d', len(shingle_vec))
        sig = self._get_sig(shingle_vec) # n-dimensional min-hash signiture
        logging.debug('got minhash sig: len(sig)=%d', len(sig))
        lsh = self._get_lsh(sig) # r-dimensional list of bucket ids
        return lsh

    def _insert_lsh(self,lsh,doc_id):
        """
        Given an LSH vector of bucket indices, this method inserts the current doc
        id in the corresponding bucket for each of the _b tables
        """
        assert doc_id not in self._seen, "Document with doc_id %d has already been inserted" % doc_id 
        accum = self._Accumulator(lsh, doc_id)
        self._seen[doc_id] = lsh if self._store_signatures else None
        if doc_id >= self._next_id:
            self._next_id = doc_id+1
        for i,band_bucket in enumerate(lsh):
            arr = self._cache[i][band_bucket]
            accum.update(arr)
            arr.append(doc_id)
        return accum.result()

    class IAccumulator(object):
        def __init__(self, lsh, doc_id):
            pass
        
        def update(self, dups):
            pass
        
        def result(self):
            raise NotImplementedError
        
    class AccumulatorDocId(IAccumulator):
        def __init__(self, lsh, doc_id):
            self._doc_id = doc_id
        
        def result(self):
            return self._doc_id
        
    class AccumulatorDups(IAccumulator):
        def __init__(self, *args):
            self._dups = set()
        
        def update(self, dups):
            self._dups.update(dups)
        
        def result(self):
            return self._dups

    @classmethod
    def _reduce_dup_buckets(cls, buckets, doc_id=None):
        # logging.debug('buckets: %s', buckets)
        all_buckets = reduce(set.update, buckets, set)
        if doc_id is not None:
            all_buckets.discard(doc_id)
        return all_buckets

    # public methods

    def get_dup_buckets(self, doc, doc_id=None):
        """
        Returns a list of buckets (which are themselves lists) that contain the ids
        of any matching documents.  If the cache was built in chronological order
        then buckets are also in chronological order
        """
        if (doc):
            lsh = self._get_lsh_from_doc(doc)
        else:
            assert self._store_signatures, "must store signatures if doc is not specified"
            assert doc_id is not None, "must specify doc or doc_id"
            lsh = self._seen[doc_id]
        for i,band_bucket in enumerate(lsh):
            yield self._cache[i][band_bucket]

    def get_dups(self, doc, doc_id=None):
        return self._reduce_dup_buckets(self.get_dup_buckets(doc, doc_id),doc_id)

    def insert(self, doc, doc_id=None):
        if doc_id is None: 
            doc_id = self._next_id
        lsh = self._get_lsh_from_doc(doc)
        logging.debug('id: %d lsh: %s', doc_id, lsh)
        return self._insert_lsh(lsh, doc_id)

    def insert_batch(self, docs):
        """Batch method for adding db docs to cache"""
        dups = list()
        logging.debug('batch inserting len(docs)=%d', len(docs) if hasattr(docs, '__len__') else -1)
        for i, doc_tuple in enumerate(docs):
            if (i % 100 == 0):
                logging.debug('batch processed %d docs', i)
            if len(doc_tuple)==2 and isinstance(doc_tuple[1], int):
                dups.append(self.insert(*doc_tuple))
            else:
                dups.append(self.insert(doc_tuple))
        return dups

    def num_docs(self):
        return len(self._seen)
    
    def max_doc_id(self):
        return self._next_id-1
    
    def num_bands(self):
        return self._b
    
    def num_rows_per_band(self):
        return self._r
    
    def num_total_rows(self):
        return self._n
    
    def shingle_len(self):
        return self._min_shingle if self._min_shingle==self._max_shingle else (self._min_shingle, self._max_shingle)

