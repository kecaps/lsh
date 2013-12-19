'''
Created on Dec 13, 2013

@author: space
'''
import unittest
import os
import sys
from nltk.metrics.distance import jaccard_distance
from math import factorial
import random

module_filename = sys.modules.get('lsh_test', sys.modules['__main__']).__file__
print os.path.join(os.path.dirname(module_filename), "..")
sys.path.append(os.path.join(os.path.dirname(module_filename), ".."))

from lsh import LSHCache, Shingler, XORHashFamily, MultiplyHashFamily

class HashFamilyTest(unittest.TestCase):
    def _test_family(self, hash_family):
        random.seed(1234)
        family = hash_family(10, 131071)
        h1 = []
        for h in family.hashn(1234):
            h1.append(h)
        self.assertEqual(10, len(h1))
        
        for ndx, h in enumerate(family.hashn(1243)):
            self.assertNotEqual(h1[ndx], h)
        
        family = hash_family(10, 131071)
        for ndx, h in enumerate(family.hashn(1234)):
            self.assertNotEqual(h1[ndx], h)
        
        random.seed(1234)
        family = hash_family(10, 131071)
        for ndx, h in enumerate(family.hashn(1234)):
            self.assertEqual(h1[ndx], h)
        
    def testXOR(self):
        self._test_family(XORHashFamily)
    
    def testMultiply(self):
        self._test_family(MultiplyHashFamily)
        
class ShinglerTest(unittest.TestCase):
    def testLenOne(self):
        s = Shingler(1)
        shingles = list(s.shingle("abcdef"))
        self.assertListEqual(map(tuple, "abcdef"), shingles)
        self.assertListEqual([('a',)], list(s.shingle("a")))
    
    def testLenTwo(self):
        s = Shingler(2)
        shingles = list(s.shingle("abcdef"))
        self.assertListEqual(map(tuple, ["ab", "bc", "cd","de", "ef"]), shingles)
        self.assertListEqual([(None,'a',)], list(s.shingle("a")))
        self.assertListEqual([('a','b',)], list(s.shingle("ab")))
    
    def testMultiLen(self):
        s = Shingler(2,3)
        shingles = set(s.shingle("abcdef"))
        self.assertSetEqual(set(map(tuple, ["ab", "bc","cd","de","ef","abc","bcd","cde","def"])),
                            shingles)
        self.assertSetEqual(set([('a','b',),('b','c',),('a','b','c',)]), set(s.shingle("abc")))
        self.assertSetEqual(set([('a','b',),(None,'a','b',)]), set(s.shingle("ab")))
        self.assertSetEqual(set([(None,'a',),(None,None,'a',)]), set(s.shingle("a")))
        
    
    def testBadArgs(self):
        with self.assertRaises(AssertionError):
            Shingler(0)
        with self.assertRaises(AssertionError):
            Shingler(2,1)
       
        
class LSHTest(unittest.TestCase):
    
    def testExample(self):
        docs = [
                "lipstick on a pig",
                "you can put lipstick on a pig",
                "you may put lipstick on a pig but it's still a pig",
                "you can put lipstick on a pig it's still a pig",
                "i think they put some lipstick on a pig but it's still a pig",
                "putting lipstick on a pig",
                "you know you can put lipstick on a pig",
                "they were going to send us binders full of women",
                "they were going to send us binders of women",
                "a b c d e f",
                "a b c d f"]

        # least strict
        random.seed(12345)
        cache = LSHCache(b=50,r=2)
        self.assertListEqual([set(),
                              set([0]),
                              set([0,1]),
                              set([0,1,2]),
                              set([0,1,2,3]),
                              set([0,1,2,3,4]),
                              set([0,1,2,3,4,5]),
                              set(),
                              set([7]),
                              set(),
                              set([9])],
                              cache.insert_batch([doc.split() for doc in docs]))

        # stricter
        random.seed(12345)
        cache = LSHCache(b=25,r=4)
        self.assertListEqual([set(),
                              set([0]),
                              set(),
                              set([1]),
                              set([2]),
                              set([0,1]),
                              set([0,1,5]),
                              set(),
                              set([7]),
                              set(),
                              set([9])],
                              cache.insert_batch([doc.split() for doc in docs]))
        # stricter still
        random.seed(12345)
        cache = LSHCache(b=20,r=5)
        self.assertListEqual([set(),
                              set([0]),
                              set(),
                              set([1]),
                              set(),
                              set([0,1]),
                              set([0,1,3,5]),
                              set(),
                              set([7]),
                              set(),
                              set([])],
                              cache.insert_batch([doc.split() for doc in docs]))
        # most strict
        random.seed(12345)
        cache = LSHCache(b=10,r=10)
        self.assertListEqual([set(),
                              set(),
                              set(),
                              set(),
                              set(),
                              set(),
                              set([1]),
                              set(),
                              set(),
                              set(),
                              set()],
                              cache.insert_batch([doc.split() for doc in docs]))


    def testPercentFound(self):
        lsh = LSHCache(b=2,r=1)
        self.assertEqual(0.75, lsh.theoretical_percent_found(0.5))
        self.assertEqual(0.96, lsh.theoretical_percent_found(0.8))
        lsh = LSHCache(b=1,r=2)
        self.assertEqual(0.25, lsh.theoretical_percent_found(0.5))
        self.assertAlmostEqual(0.64, lsh.theoretical_percent_found(0.8))
        lsh = LSHCache(b=10,r=10)
        self.assertAlmostEqual(0.0097, lsh.theoretical_percent_found(0.5), places=4)
        self.assertAlmostEqual(0.6789, lsh.theoretical_percent_found(0.8), places=4)
        lsh = LSHCache(b=20,r=5)
        self.assertAlmostEqual(0.4701, lsh.theoretical_percent_found(0.5), places=4)
        self.assertAlmostEqual(0.9996, lsh.theoretical_percent_found(0.8), places=4)
        lsh = LSHCache(b=25,r=4)
        self.assertAlmostEqual(0.8008, lsh.theoretical_percent_found(0.5), places=4)
        self.assertAlmostEqual(1.0000, lsh.theoretical_percent_found(0.8), places=4)
        
    def testLSH(self):
        strings = [
                   "abcdefghijklmnopqrstuvwxyz",
                   "abcdefghijklmnopqrstuvw",
                   "defghijklmnopqrstuvw",
                   "zyxwvutsrqponmlkjihgfedcba",
                   "1abcdefghijklmnopuvw1",
                   "123456789",
                   "012345678",
                   "234567890",
                   ]
        for i, a in enumerate(strings):
            for j, b in enumerate(strings[i+1:]):
                print "'%s' (%d) <=> (%d)'%s': %f" % (a, i,j+i+1, b, 1-jaccard_distance(set(a),set(b)))

        random.seed(12345)
        lsh = LSHCache(shingler=Shingler(1))
        self.assertListEqual([set(),
                              set([0]),
                              set([0,1]),
                              set([0,1,2]),
                              set([0,1,2,3]),
                              set(),
                              set([5]),
                              set([5,6])], lsh.insert_batch(strings))
        
    def testBadArgs(self):
        with self.assertRaises(AssertionError):
            LSHCache(b=10, r=7, n=100)
        with self.assertRaises(AssertionError):
            # prime number of rows
            LSHCache(n=101)
        with self.assertRaises(AssertionError):
            LSHCache(n=100, r=7)
        with self.assertRaises(AssertionError):
            LSHCache(n=100, b=7)
        
    def testLSHArgs(self):
        lsh = LSHCache()
        self.assertEqual(20, lsh.num_bands())
        self.assertEqual(5, lsh.num_rows_per_band())
        self.assertEqual(100, lsh.num_total_rows())
        self.assertEqual(2, lsh.shingler().shingle_len())
        
        lsh = LSHCache(b=10, r=7)
        self.assertEqual(10, lsh.num_bands())
        self.assertEqual(7, lsh.num_rows_per_band())
        self.assertEqual(70, lsh.num_total_rows())
        self.assertEqual(2, lsh.shingler().shingle_len())
        
        lsh = LSHCache(n=70, r=7)
        self.assertEqual(10, lsh.num_bands())
        self.assertEqual(7, lsh.num_rows_per_band())
        self.assertEqual(70, lsh.num_total_rows())
        self.assertEqual(2, lsh.shingler().shingle_len())
        
        lsh = LSHCache(n=70, b=10)
        self.assertEqual(10, lsh.num_bands())
        self.assertEqual(7, lsh.num_rows_per_band())
        self.assertEqual(70, lsh.num_total_rows())
        self.assertEqual(2, lsh.shingler().shingle_len())

        lsh = LSHCache(n=70, b=10, r=7)
        self.assertEqual(10, lsh.num_bands())
        self.assertEqual(7, lsh.num_rows_per_band())
        self.assertEqual(70, lsh.num_total_rows())
        self.assertEqual(2, lsh.shingler().shingle_len())
        
        lsh = LSHCache(shingler=Shingler(5))
        self.assertEqual(20, lsh.num_bands())
        self.assertEqual(5, lsh.num_rows_per_band())
        self.assertEqual(100, lsh.num_total_rows())
        self.assertEqual(5, lsh.shingler().shingle_len())
        
        lsh = LSHCache(shingler=Shingler(2,3))
        self.assertEqual(20, lsh.num_bands())
        self.assertEqual(5, lsh.num_rows_per_band())
        self.assertEqual(100, lsh.num_total_rows())
        self.assertEqual((2,3,), lsh.shingler().shingle_len())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLSHCreation']
    unittest.main()