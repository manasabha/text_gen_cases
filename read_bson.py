from bson import decode_file_iter
from gensim.models import Word2Vec
import os
from tqdm import tqdm

def mongodoc2sentences(item, outfolder):
    try:
        passages = [passage['text'] for passage in item['passages']]
        out = '\n'.join(passages)
        fn = item['citation'].replace(' ', '_').replace('/', '_')
        # print 'write to file {}'.format(fn)
        outpath = os.path.join(outfolder, fn)
        if not os.path.exists(outpath):
            with open(outpath, 'w') as f:
                f.write(out.encode('utf-8'))
        return passages
    except:
        print('Bad document {}'.format(item['citation']))

class Sentences(object):
    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __iter__(self):
        for item in tqdm(decode_file_iter(open(self.fn, 'rb'))):
            sentences = mongodoc2sentences(item, '../data/cases')
            if sentences:
                for sentence in sentences:
                    yield sentence.split()
                self.count+=1

if __name__=="__main__":
    output_folder = '../data/cases'
    sentenceIterator = Sentences('/Users/engineer/Desktop/development/document.bson')
    w2vmodel = Word2Vec(sentenceIterator, size=200, window=10, min_count=100, workers=4)
    w2vmodel.save('../data/models/casesw2vsize200win10')
    print 'done'