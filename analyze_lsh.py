'''
Created on Dec 13, 2013

@author: space
'''
import argparse
import logging
import random
import itertools as it
import functools as ft
from lsh import LSHCache, XORHashFamily, MultiplyHashFamily, Shingler
from nltk.metrics.distance import jaccard_distance, masi_distance, edit_distance

minhash_choices = { 'xor': XORHashFamily,
                    'multiply': MultiplyHashFamily,
                  }

similarity_choices = { 'jaccard': lambda a,b,s: 1 - jaccard_distance(set(s.shingle(a)), set(s.shingle(b))),
                       'masi': lambda a,b,s: 1 - masi_distance(set(s.shingle(a)), set(s.shingle(b))),
                       'edit': lambda a,b,s: 1 - float(edit_distance(a,b))/max(len(a),len(b)),
                       'edit_transposition': lambda a,b,s: 1-float(edit_distance(a,b,True))/max(len(a),len(b)) }

generator_choices = { 'combinations': it.combinations,
                      'combinations_replacement': it.combinations_with_replacement,
                      'permutations': it.permutations }

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze performance of LSH over a mock generated data set")

    lsh_group = parser.add_argument_group('LSH Cache parameters')
    
    lsh_group.add_argument("-b", "--num-bands", type=int,
                        help="""number of bands in LSH cache""")
    lsh_group.add_argument("-r", "--num-rows", type=int, 
                        help="""number of rows per band in LSH cache""")
    lsh_group.add_argument("-n", "--num-total", type=int,
                        help="""total number of rows in LSH cache""")
    lsh_group.add_argument("--minhash",
                           choices=minhash_choices.keys())
    lsh_group.add_argument("-u", "--universe-size", type=int,
                           help="""size of the shingle universe"""),
    lsh_group.add_argument("--shingle-len", default=[], nargs='*', type=int,
                        help="""length of shingles generated form document""")

    doc_group = parser.add_argument_group('Document Generation parameters')
    doc_group.add_argument("-d", "--num-docs", type=int,
                        help='''number of documents to generate''')
    doc_group.add_argument("--doc-len", type=int, default=[10], nargs='+',
                         help='''length of generated documents''')
    doc_group.add_argument("-t", "--num-tokens", default=10, type=int,
                        help='''number of tokens used in generated documents''')
    doc_group.add_argument("-g", "--generator", default='combinations',
                           choices=generator_choices.keys(),
                           help='''how to generate the documents from the set of tokens and the document length''')
#    doc_group.add_argument("--token-distribution", default=['uniform'],
#                        choices=['uniform','normal','zipf','mandelbrot'],
#                        help='''distribution to use for choosing tokens for documents.  Defaults to %(default)s''')
#    doc_group.add_argument("--token-distribution-args", type=float, default=[], nargs='*',
#                           help='''arguments used to describe given distribution (e.g., mu and sd for normal distribution)''')
 
    doc_group.add_argument("-s", "--similarity",  default='jaccard',
                           choices=similarity_choices.keys(),
                           help='''similarity algorithm to use to measure distance between documents.  Defaults to %(default)s''')
    parser.add_argument('--sim-cuts', default=10, type=int,
                        help='''cuts of similarity range [0-1] in which to report counts of real similar and lsh similar objects''')
    parser.add_argument('--log', default='info',
                        choices=('debug','info','warning','error','critical'),
                        help='level of logging to capture')
    parser.add_argument('--seed', type=long, required=False, default=[], nargs='*',
                        help='''set random number generator seed.  If specified, the seed will be set before the creation of the LSH cache and then
                                before creation of documents.  Multiple seeds can be specified in which case the seeds will be used sequentially''')
    args = parser.parse_args(argv)
    
    logging.basicConfig(level=getattr(logging, args.log.upper()),
                        datefmt='%H:%M:%S', format='[%(asctime)s] %(levelname)s %(message)s')
    return args

def seed_from_args(args):
    if args.seed:
        seed = args.seed.pop()
        random.seed(seed)
        # np.random.seed(seed)
        args.seed.append(seed)

def lsh_cache_from_args(args):
    seed_from_args(args)
    kwargs = {"shingler": Shingler(*args.shingle_len) }
    if args.minhash:
        kwargs['minhash'] = minhash_choices[args.minhash]
    for arg_key, kwarg_key in (('num_total','n'),('num_bands','b'),('num_rows','r'),('universe_size',)*2):
        value = getattr(args, arg_key)
        if value:
            kwargs[kwarg_key] = value
    cache = LSHCache(**kwargs)
    # logging.info(str(cache))
    return cache

def tokens_from_args(args):
    return range(1,args.num_tokens+1)

def doc_len_from_args(args):
    if len(args.doc_len) == 2:
        logging.info("generating documents with length between %d and %d",*args.doc_len)
        return xrange(args.doc_len[0], args.doc_len[1]+1)
    else:
        logging.info("generating documents with length %s", args.doc_len)
        if args.doc_len[0] == 0:
            args.doc_len.pop(0)
        return iter(args.doc_len)

def doc_generator_from_args(args):
    doc_len = doc_len_from_args(args)
    logging.info("generating documents made of %d tokens by %s", 
                 args.num_tokens, args.generator)
    generator = generator_choices[args.generator]
    def gen_doc():
        return it.chain.from_iterable(it.imap(ft.partial(generator, tokens_from_args(args)), doc_len))
    return gen_doc

def similar_from_args(args):
    logging.info("calculating similarity with %s", args.similarity)
    return similarity_choices[args.similarity]

def main(argv=None):
    args = parse_args()
    cache = lsh_cache_from_args(args)
    gen_doc = doc_generator_from_args(args)
    calc_similar = similar_from_args(args)
    
    total_distribution = [0]*(args.sim_cuts+1)
    lsh_distribution = [0]*(args.sim_cuts+1)
    docs = []
    
    seed_from_args(args)
    try:
        for ndx, doc in enumerate(gen_doc()):
            if args.num_docs and ndx == args.num_docs:
                break
            if ndx and ndx % 100 == 0:
                logging.info("processed %d documents, %d comparisons", ndx, sum(total_distribution))
            docs.append(doc)
            logging.debug("%d: %s", ndx, doc)
            lsh_similar = cache.insert(doc)
            for o_ndx in xrange(ndx):
                sim_ndx = int(args.sim_cuts*calc_similar(doc, docs[o_ndx], cache.shingler()))
                total_distribution[sim_ndx] += 1
                if o_ndx in lsh_similar:
                    lsh_distribution[sim_ndx] += 1
        else:
            logging.info('done!')
    except KeyboardInterrupt:
        logging.warn("Received keyboard interrupt.  Stopping generation and comparisons of documents")
    
    logging.info('processed %d documents, %d comparisons', len(docs), sum(total_distribution))

    print ("| %12s "*5+'|') % ("Similarity", "LSH Count", "Total Count", "% in LSH", "Theoretical %")
    print "|" + ("-"*14+'+')*4 + "-"*14 + "|"
    total_theoretical_pct = 0
    for i, (lsh_count, total_count) in enumerate(zip(lsh_distribution, total_distribution)):
        sim = float(i)/args.sim_cuts
        pct_in_lsh = float(lsh_count)/total_count if total_count else float('inf')
        theoretical_pct = cache.theoretical_percent_found(sim)
        print "| %12.2f | %12d | %12d | %12.4f | %12.4f |" % \
            (sim, lsh_count, total_count, pct_in_lsh, theoretical_pct)
        total_theoretical_pct += theoretical_pct * total_count 
    print "|" + ("="*14+'+')*4 + "="*14 + "|"
    print "| %12s | %12d | %12d | %12.4f | %12.4f |" % \
            ("Total", sum(lsh_distribution), sum(total_distribution),
             float(sum(lsh_distribution))/sum(total_distribution), total_theoretical_pct/sum(total_distribution))  

if __name__ == '__main__':
    main()
    