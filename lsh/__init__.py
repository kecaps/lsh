from collections import defaultdict
import itertools as it
import random
import sys
import time
import logging
from math import sqrt
import inspect

logging.getLogger().setLevel(logging.INFO)


class Shingler(object):
    """
    Handles turning a document (a list of tokens) into a sequence of shingles to 
    be used for minhashing
    """
    
    def __init__(self, shingle_len=2, max_shingle=None):
        """
        Create a shingler by specifying the length of shingle.  If you want shingles of 
        multiple length, then provide a minimum and maximum shingle length.
        Defaults to a shingle length of 2
        """
        assert shingle_len > 0, 'shingle length must be greater than 0, not %d' % shingle_len
        self._begin_shingle = shingle_len
        if max_shingle:
            assert max_shingle >= shingle_len, 'max_shingle must be greater than shingle_len'
            self._end_shingle = max_shingle + 1
        else:
            self._end_shingle = shingle_len + 1
    
    def shingle_universe(self, token_universe):
        """
        Given a size of the token universe, return the size of the shingle universe
        """
        return sum(map(lambda sl: token_universe**sl, xrange(self._begin_shingle, self._end_shingle)))
   
    def shingle_len(self):
        """
        return the shingle length being used. If only a single shingle length is used, then a single item is returned
        If multiple shingle lengths are used a tuple of the beginning and end (inclusive) is used.
        """
        return self._begin_shingle if not self.is_multi_shingler() else (self._begin_shingle, self._end_shingle-1,)
 
    def shingle_len_str(self):
        """
        return a string representing the shingle length.  If there is only one shingle length, it is simply the number
        Otherwise it represents the range of shingle lengths as 'lower<=upper'
        """
        return "<=".join(sorted(set((self._begin_shingle, self._end_shingle-1))))

    def is_multi_shingler(self):
        """
        indicates whether this shingler creates shingles of multiple lengths (i.e., whether it shingles the document multiple
        times)
        """
        return self._begin_shingle+1 != self._end_shingle
    
    def shingle(self, doc):
        """
        Takes a document (a list of tokens) and  of tokenized words returns a sequence and maps each shingle to a unique id.
        These unique ids, are then added to the shingle_vec object which is just a sparse
        vector implemented as a dict with v[id]=1 when a shingle id is present
        """
        for n in xrange(self._begin_shingle, self._end_shingle):
            if len(doc) < n:
                yield (None,)*(n-len(doc)) + tuple(doc)
            else:
                for j in xrange(len(doc) - (n-1)):
                    yield tuple(doc[j:j+n])

    def __str__(self):
        return ("Shingler(len %d<=%d)" if self.is_multi_shingler() else "Shingler(len %d)") % self.shingle_len()                    
        
class IHashFamily(object):
    """
    An interface for a hash family provider.  It provides a series of random hashes
    from a universal hash family.  This can then be used for minhashing.
    """
    
    def __init__(self, num_hashes, num_buckets):
        """
        Initialize the hash family by indicating how many hashes are needed.  
        Also indicate the number of buckets that will be hashed to (if that is necessary
        for choosing parameters).  The hash function is not required to return values less
        than num_buckets (They will be modulo'd afterwards) 
        """
        pass
    
    def hashn(self, x):
        """
        return a sequence of n hashes of the value x.  n is provided in the construction
        of the hash family
        """
        raise NotImplementedError()
        
class XORHashFamily(IHashFamily):
    """
    An implementation of a hash family.  This uses random 32-bit hash values which are
    xor'd with the value (It assumes that the value is an integer)
    """
    
    def __init__(self, num_hashes, num_buckets):
        """
        Initialize a random number of 32-bit fields for xoring
        """
        self._memomask = [ int(random.getrandbits(32)) for _ in xrange(num_hashes)]

    def _xor_hash(self,x, mask):
        """
        This is a simple hash function which returns the result of a bitwise XOR
        on the input x and the 32-bit random mask
        """
        return int(x ^ mask)

    def hashn(self, x):
        """ 
        generate the series of hashes of the value to be used for finding the minhash
        The implementation uses _xor_hashing with a series of random 32-bit fields
        """
        # trim x to 32-bits
        x = x & 0xffffffff
        return it.imap(lambda mask: self._xor_hash(x, mask), self._memomask)
 
class MultiplyHashFamily(IHashFamily):
    """
    An implementation of a hash family that uses random multiplication of the
    form a * (x>>4) + b * x + c.
    It assumes that the value is an integer.
    This method was described in an exercise (http://www.cs.uoi.gr/~tsap/teaching/2012f-cs059/assignments/assignment2-en.pdf)
    and implemented in java (http://blogs.msdn.com/b/spt/archive/2008/06/10/set-similarity-and-min-hash.aspx)
    """
     
    def __init__(self, num_hashes, num_buckets):
        """
        Initialize a set of 3 random integers < num_buckets for each hash
        """
        self._params = [ [random.randint(1,num_buckets) for _ in xrange(3)] for _ in xrange(num_hashes)]
    
    def _mult_hash(self, x, params):
        return params[0]*(x>>4) + params[1]*x + params[2]
    
    def hashn(self, x):
        return it.imap(lambda params: self._mult_hash(x, params), self._params)

    
class LSHCache:
    """
    Locality-Sensitive Hashing (LSH) implementation as described in 
    Mining of Massive Dataset, Chapter 3 by Anand Rajaraman and Jeff Ullman, Chapter 3
    http://infolab.stanford.edu/~ullman/mmds/ch3.pdf
    
    A document is a sequence of items which will be shingled together to look for
    similarity.

    This can bep
    This can be sub-classed to allow for implementing your own method for hashing shingles
    or hashing a shingle into the set of hashes used by minhash.
    
    Override _minhash_hash and _minhash_init in order to change how minhashes are calculated
    
    Override _hash_shingle in order to change how shingles are hashed. The default behavior
    is to simply hash each tuple.  Another approach would be to keep a cache mapping each shingle
    seen to a unique id and return that.
    """
    
    def __init__(self, b=None, r=None, n=None,
                 shingler=Shingler(2), shingle_hash=hash,
                 universe_size = 131071, minhash=MultiplyHashFamily,
                 store_signatures = False, dups_on_insert = True):
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
            
        You may also specify how a document is shingled and how that shingle is hashed.  Those arguments are:
            shingler: an instance of Shingler (has the method shingle(doc) that returns an iteration)
            shingle_hash: how to hash the shingles returned by the shingler.  Defaults to built-in hash

        You may also specify what method is used for minhashing.  
            universe_size:  size of token universe.  If you know the number of possible tokens
                            in your universe, you can specify it so that hashing better simulates
                            a random permutation of rows in a num_tokens x num_rows matrix.  
                            If it is not known, it is better to leave as a prime number for better 
                            hash performance.  Defaults to 131071
            minhash:        class that implements IHashFamily interface or a method that takes a single
                            argument and returns a sequence of n hashes

        Finally, there are a few additional optional arguments.
            store_signatures: whether to store the generated signatures.  This allows later lookups to
                              be done by doc_id rather than having to rehash a document in the cache. By default,
                              signatures are not stored
            dups_on_insert:   whether to return the duplicates found in the cache when a new document is
                              inserted.  If False, it returns the generated doc_id for the inserted document.
                              By default, True
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

        logging.debug("building LSH cache with %d total rows (%d bands, %d rows per band) with %s and hashing with %s",
                      n, b, r, shingler, minhash)

        if inspect.isclass(minhash):
            minhash = minhash(n, universe_size).hashn

        # assign it
        self._b = b
        self._r = r
        self._n = n
        self._shingler = shingler
        self._shingle_hash = shingle_hash
        self._minhash = minhash
        self._universe_size = universe_size
        self._store_signatures = store_signatures
        self._Accumulator = self.AccumulatorDups if dups_on_insert else self.AccumulatorDocId

        # make it 
        self._seen = {} # the set of doc ids which have already been hashed
        self._next_id = 0
        self._cache = [defaultdict(list) for _ in xrange(self._b)]

    def _hash_shingle(self, shingle):
        return hash(shingle)
        
    def _get_shingle_vec(self, doc):
        """
        Takes a sequence of tokenized words and maps each shingle to a unique id.
        These unique ids, are then added to the shingle_vec object which is just a sparse
        vector implemented as a dict with v[id]=1 when a shingle id is present
        """
        return set(it.imap(lambda shingle: self._shingle_hash(shingle) % self._universe_size, 
                           self._shingler.shingle(doc)))

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
            for i,h in enumerate(self._minhash(shingle)):
                h  = h % self._universe_size
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
        shingle_vec = self._get_shingle_vec(doc)
        sig = self._get_sig(shingle_vec) # n-dimensional min-hash signiture
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
        dups = []
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
    
    def shingler(self):
        return self._shingler
    
    def theoretical_percent_found(self, pct_similar):
        """
        Returns the theoretical percentage of documents that should be found with the
        given pct_similarity from this cache
        """
        return 1 - (1-pct_similar**self._r)**self._b
    

