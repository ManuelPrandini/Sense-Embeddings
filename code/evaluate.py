import string
import re
from tqdm import tqdm
from scipy import spatial, stats
import logging
import preprocess as pre
from gensim.models import KeyedVectors


#Define some paths
path_combined = "../resources/combined.tab"
path_embeddings = "../resources/embeddings.vec"

'''
Method that calculates the highest score basing on cosine similarity
between two sets of sense embeddings from two words.
:path the file where take the word and the human score 
:sense_embeddings the vocab contained the sense_embeddings
:model the word2Vec model used to verify the score between pair of words
:return the gold score set and the cosine similarity score set 
'''
def word_similarity(path,sense_embeddings,model):
    i = 0
    gold = []
    cosines = []
    with open(path,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            if(i==0):
                i+=1
                continue
            else:
                line = line.lower().split("\t")
                
                if line[0] in sense_embeddings:
                    s1 = sense_embeddings[line[0]]
                if line[1] in sense_embeddings:
                    s2 = sense_embeddings[line[1]]
                gold.append(float(line[2].strip()))
                score = -1.0
                for sense1 in s1:
                    for sense2 in s2:
                        if sense1 in model.wv.vocab and sense2 in model.wv.vocab:
                            score = max(score,model.wv.similarity(sense1,sense2))
                cosines.append(score)
        f.close()
    return gold, cosines




def main():
    #upload the the sense embeddings vocab
    print("Load the sense embeddings vocab...")
    sense_embeddings = pre.load_sense_embeddings_vocab()
    print("Done!")
    print("Load the Word2Vec model ...")
    model = KeyedVectors.load_word2vec_format(path_embeddings,binary=False)
    print("Done!")
    print("Calculate Spearman..")
    gold, cosines = word_similarity(path_combined,sense_embeddings,model)
    print(stats.spearmanr(gold,cosines))

if __name__ == "__main__":
    main()
