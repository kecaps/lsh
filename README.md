lsh
===

a pure python locality senstive hashing implementation

## Installation
`lsh` is packaged with setuptools so it can be easily installed with pip like this:

````
$ cd lsh/
$ [sudo] pip install -e .
````

## Usage
```python

from lsh import LSHCache

cache = LSHCache()
    
docs = [
  "lipstick on a pig",
  "you can put lipstick on a pig",
  "you    can put lipstick on a pig but it's still a pig",
  "you can put lipstick on a pig it's still a pig",
  "i think they put some lipstick on a pig but it's still a pig",
  "putting lipstick on a pig",
  "you know you can put lipstick on a pig",
  "they were going to send us binders full of women",
  "they were going to send us binders of women",
  "a b c d e f",
  "a b c d f"]

dups = {}
for i, doc in enumerate(docs):
    dups[i] = cache.insert(doc.split(), i)
    ...
````

## Tools

`analyze_lsh` will generate a model set of documents and run lsh over it.  It will generate statistics
showing how many documents a cache will return based on the measured similarity of the documents. e.g.,

````
$ python analyze_lsh.py --doc-len 1 10 --num-tokens 10 --similarity jaccard 
|   Similarity |    LSH Count |  Total Count |     % in LSH | Theoretical % |
|--------------+--------------+--------------+--------------+--------------|
|         0.00 |            0 |       301195 |       0.0000 |       0.0000 |
|         0.10 |           62 |       130972 |       0.0005 |       0.0002 |
|         0.20 |          981 |        44804 |       0.0219 |       0.0064 |
|         0.30 |         2611 |        22244 |       0.1174 |       0.0475 |
|         0.40 |         2849 |        10834 |       0.2630 |       0.1860 |
|         0.50 |         4591 |         8126 |       0.5650 |       0.4701 |
|         0.60 |         2601 |         2958 |       0.8793 |       0.8019 |
|         0.70 |          836 |          848 |       0.9858 |       0.9748 |
|         0.80 |          771 |          772 |       0.9987 |       0.9996 |
|         0.90 |            0 |            0 |          inf |       1.0000 |
|         1.00 |            0 |            0 |          inf |       1.0000 |
|==============+==============+==============+==============+==============|
|        Total |        15302 |       522753 |       0.0293 |       0.0214 |
````

## Roadmap
* add more tests
* add `save()` and `from_file()` methods
* rewrite with redis backend?
