import string
import re
from nltk.corpus import stopwords
from tqdm import tqdm
import logging
from nltk.stem import PorterStemmer

#Path for the sense_embeddings_vocab
path_sense_embeddings_vocab = "../resources/sense_embeddings_vocab.txt"
'''
Method that taken in input a file, clean it removing some data mistakes,
punctuation, and do the stemmatization. Then prepare the sentences to give in input at the w2v model
:file the path of the file to clean
:return an array with splitted sentences 
'''
def clean_data_file(file):
    # Load stop words
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    result = []
    ps = PorterStemmer()
    with open(file,'r',encoding="utf-8") as f:
        for line in tqdm(f):
            line = re.split("[\s|\.]",line.lower())
            #Remove from the words some end points
            line = [w[:-1] if w.endswith(".") else w for w in line ]
            #Remove some multiple points
            line = [w for w in line if not re.match(r'\.+',w)]
            #Remove the &apos html
            line = [w.replace("&apos;","") if w.startswith("&apos;") else w for w in line ]
            #Remove from the words the initial ' character
            line = [w[1:] if w.startswith("'") else w for w in line ]
            #Remove no printable ASCII characters from TOM
            line = ['' if len(w)<2 and w not in string.printable else w for w in line]
            #Remove punctuation
            line = [w for w in line if w not in string.punctuation ]
            # Remove stop words
            line = [ps.stem(w) for w in line if w not in stop_words]
            result.append(line)
        f.close()
    return result

'''
Method that create the sense_embeddings_vocab through
a sentences array and save it into a file.
:sentence_array the array contained each sentence to process
'''
def make_sense_embeddings_vocab(sentence_array):
    result = {}
    for line in tqdm(sentence_array):
        for word in line:
            if "_bn:" in word:
                split = word.split("_bn:")[0]
                if( split not in result):
                    result[split] = set()
                    result[split].add(word)
                else:
                    result[split].add(word)
    with open(path_sense_embeddings_vocab,'w',encoding='utf8') as f:
        for k, v in result.items():
            f.write(k)
            for i in v:
                f.write(" "+i)
            f.write("\n")
        f.close()

'''
Method that load the sense_embeddings_vocab file and insert it into
a new dict.
This method can be invoked only if it was invoked before make_sense_embeddings_vocab.
:return the dict with the sense_embeddings_vocab
'''
def load_sense_embeddings_vocab():
    result = dict()
    with open(path_sense_embeddings_vocab,'r',encoding='utf-8') as f:
        for line in f:
            splitted = line.split()
            result[splitted[0]] = set(splitted[1:])
        f.close()
    return result