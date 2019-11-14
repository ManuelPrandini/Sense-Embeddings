import string
import re
from gensim import models
from tqdm import tqdm
import psutil
import logging
import preprocess as pre

#Define some paths
path_precision_result = "../resources/precision_train_x.txt"
path_tom_result = "../resources/tom_sentences_train_x.txt"
path_embeddings_dirty = "../resources/embeddings_dirty.vec"
path_embeddings_clear = "../resources/embeddings.vec"
path_weights = "../resources/weights.model"

'''
Method that create the W2V model and train it.
:sentence_array the splitted array contained the input words of the word2Vec model
:size the size of the feature vector
:min_count  
'''
def createW2VModel(sentence_array,size,min_count,window,hs,iter,alpha,sample):
    cores = psutil.cpu_count()

    #output for the execution
    logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )

    model = models.Word2Vec(sentence_array,size=size,
                            min_count=min_count,
                            window=window,
                            hs = hs,
                            iter=iter,
                            alpha=alpha,
                            sample=sample,
                            workers=cores)
    return model

'''
Method used to filter the embeddings and remove all the words that
not contain the synset.
:path the path of the embedding file dirty
:array_size the size of the feature vector
'''
def clean_embeddings(path,array_size):
    number_of_synsets = 0
    with open(path_embeddings_clear,'w',encoding='utf-8') as fw:
        with open(path,'r',encoding='utf-8') as fr:
            for line in tqdm(fr):
                line = line.split()
                if("_bn:" in line[0]):
                    number_of_synsets+=1
                    fw.write(line[0])
                    for v in line[1:]:
                        fw.write(" "+v)
                    fw.write("\n")
            fw.seek(0)
            fw.write(str(number_of_synsets)+" "+str(array_size)+"\n")
            fr.close()
        fw.close()

def main():
    #CLEAN THE DATA
    print("Clean eurosense high precision data...")
    sentence_array = pre.clean_data_file(path_precision_result)
    print("Done!")
    print("Clean tom data...")
    sentence_tom = pre.clean_data_file(path_tom_result)
    print("Done!")
    sentence_array += sentence_tom

    #MAKE THE SENSE_EMBEDDINGS_VOCAB
    print("Generate sense embeddings vocab...")
    pre.make_sense_embeddings_vocab(sentence_array)
    print("Done!")

    #DEFINE THE W2V MODEL
    print("Create and train the Word2Vec model...")
    model = createW2VModel(sentence_array,300,2,5,1,10,0.025,1e-3)
    print("Model created!")
    model.wv.save_word2vec_format(path_embeddings_dirty,binary=False)
    print("Clean the embeddings leaving only the synsets...")
    clean_embeddings(path_embeddings_dirty,300)
    print("Done!")
    print("Model saved in ",path_embeddings_clear)



if __name__ == "__main__":
    main()