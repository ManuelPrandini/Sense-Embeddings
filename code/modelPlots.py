import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import re


'''
Method that create a sub dictionary with only the relative synsets of the
specific word passed in input.
:model the w2v model where take the embeddings
:word word used to filter on the model.vocab
'''
def get_all_synsets(model,word):
    same_synset = {}
    for k, v in model.vocab.items():
        if re.match(word+r'_bn:[0-9]{8}[a|r|v|n]$',k)!=None:
            same_synset[k] = v
    return same_synset


'''
Method used to plot the TSNE model synsets for a specific word passed in input or
their similar synsets take a synset.
:model the w2v model
:w_to_search the word to search in the embeddings
:modality if set to "synsets" , the method plot only the synsets relative
to the w_to_search, while if set to "similar", it plot also the synsets 
most similar to the relative synsets.
'''
def tsne_plot(model,w_to_search,modality="synsets"):
    "Creates a TSNE model and plots it"
    labels = []
    tokens = []
    sub_tokens = []
    sub_labels = []
    #take relative synsets
    synsets = get_all_synsets(model,w_to_search)
    if modality == "synsets":
        for word in synsets:
            sub_tokens.append(model[word])
            sub_labels.append(word)
        tokens.append(sub_tokens)
        labels.append(sub_labels)

    elif modality == "similar":
        for word in synsets:
            sub_tokens.append(model[word])
            sub_labels.append(word)
            for sim in model.wv.most_similar(word):
                sub_tokens.append(model[sim[0]])
                sub_labels.append(sim[0])
            tokens.append(sub_tokens)
            labels.append(sub_labels)
            sub_labels = []
            sub_tokens = []
    
    plt.figure(figsize=(25, 25),) 
    plt.title("Synsets for "+w_to_search)
    colors = {0:"red",1:"orange",2:"yellow",3:"blue",4:"pink",5:"green",6:"brown",7:"lime",
            8:"cyan",9:"purple",10:"teal"}
    for t in range(len(tokens)):
        
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
        new_values = tsne_model.fit_transform(tokens[t])

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])


        for i in range(len(x)):
            #set color
            if modality == "synsets":
                color = i
            else:
                color = t
                    
            plt.scatter(x[i],y[i],c=colors[color],s=20*4)
            plt.annotate(labels[t][i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                        size = 20,
                         ha='right',
                         va='bottom')
    plt.savefig("../resources/"+w_to_search+"_synsets.png")
    plt.show()

'''
Method that plot the clusters of the word passed in input
:model the word2vec model
:word_array the set of synset to pass in input
:topn how many word most similar have to be present in the plot for each
word passed in input
:title the title of the plot
'''
def tsne_plot_cluster(model,word_array,topn,title):
    "Creates a TSNE model and plots it"
    labels = []
    tokens = []
    sub_tokens = []
    sub_labels = []
    #take relative synsets
    for w in word_array:
        sub_tokens.append(model[w])
        sub_labels.append(w)
        for sim in model.wv.most_similar(w,topn=topn):
            sub_tokens.append(model[sim[0]])
            sub_labels.append(sim[0])
        tokens.append(sub_tokens)
        labels.append(sub_labels)
        sub_labels = []
        sub_tokens = []

    
    plt.figure(figsize=(16, 9),) 
    plt.title(title,fontsize=20)
    colors = {0:"red",1:"blue",2:"yellow",3:"orange",4:"pink",5:"green",6:"brown",7:"lime",
            8:"cyan",9:"purple",10:"teal"}
    for t in range(len(tokens)):
        
        tsne_model = TSNE(perplexity=10,n_components=2, n_iter=2000,learning_rate=10.0)
        new_values = tsne_model.fit_transform(tokens[t])

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])


        for i in range(len(x)):
                    
            plt.scatter(x[i],y[i],c=colors[t],s=20*4)
            plt.annotate(labels[t][i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                        size = 16,
                         ha='right',
                         va='bottom')
    plt.savefig("../resources/"+title+"_synsets_cluster.png")
    plt.show()

'''
Method that plot the histogram of the similar synsets of the synset passed
in input
:model the word2vec model
:synset the synset to pass in input to obtain the similar synsets
'''
def plot_similar_histogram(model,synset):
    x = []
    label = []
    sim = model.wv.most_similar(synset)
    for l, v in sim:
        label.append(l)
        x.append(v)
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(x)),x)
    plt.xticks(np.arange(len(x)), label,rotation=90,fontsize=15)
    plt.ylabel("cosine similarity",fontsize=15)
    plt.title("Most similar synsets of "+synset,fontsize=15)
    plt.savefig("../resources/"+synset+"_most_similar.png")
    plt.show()
