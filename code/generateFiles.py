from lxml import etree as ET
import string
import re
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import logging


path_precision= "../resources/eurosense.v1.0.high-precision/EuroSense/eurosense.v1.0.high-precision.xml"
path_mapping = "../resources/bn2wn_mapping.txt"
path_precision_result = "../resources/precision_train_x.txt"
path_tom = "../train-o-matic-data/train-o-matic-data/EN/EN.500-2.0/mergedFiles/evaluation-framework-ims-training.xml"
path_tom_result = "../resources/tom_sentences_train_x.txt"

'''
Method used to create the dict from the file
that contains the mapping from babelnet to wordnet
and the inverse dict, from wordnet to babelnet.
:file path file taht contains the bn2wn mapping
:return the two dictionaries, one from bn to wn and the other
viceversa.
'''
def make_dict_mapping(file):
    babel_mapping = {}
    wordnet_mapping = {}
    #Read the mapping file
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            splitted_line = line.split()
            babel_mapping[splitted_line[0]] = splitted_line[1]
            wordnet_mapping[splitted_line[1]] = splitted_line[0]
        f.close()
    return babel_mapping, wordnet_mapping


'''
Method used to create the file with sentences from
the eurosense xml file.
:path the path of the xml file
:bn2wn_mapping the dict that contains the babelnet to wordnet mapping
:file the path where write the sentences take from the xml file
'''
def make_sentences_file_eurosense(path,bn2wn_mapping,file):
    annotations = dict()
    get_sentence = None
    with open(file,'w',encoding='utf-8') as f:
        for event, elem in tqdm(ET.iterparse(path, events={'start','end'})):
            if event == 'start':
                #take the sentence in text
                if elem.tag == 'text' and elem.attrib['lang'] == 'en' and type(elem.text) == str:
                    result = " "+elem.text+" "
                    get_sentence = True
                #take the annotations
                if elem.tag == 'annotation' and elem.attrib['lang'] == 'en' and type(elem.text) == str:
                    annotations[elem.attrib['anchor']] = elem.attrib['lemma'].replace(" ","_")+"|"+elem.text

            if event == 'end':
                if elem.tag == 'sentence' and get_sentence:
                    for k in sorted(annotations,key=len,reverse=True):
                        value = annotations[k].split("|")
                        #check if in bn2wn mapping
                        if value[1] in bn2wn_mapping:
                            #check if the synset format is correct
                            if(re.match(r"\s?[A-Z|a-z][a-z]*_?[A-Z|a-z][a-z]*\_bn\:[0-9]{8}[a|v|n|r]\s?$",value[0]+"_"+value[1]) != None):
                                result = result.replace(" "+k+" "," "+value[0]+"_"+value[1]+" ")
                    
                    f.write(result.lower()+"\n")
                    get_sentence = False
                    elem.clear()
                    annotations = dict()
    f.close()


'''
Method used to create the file with sentences from
the Tom xml file. The synset to replace with the correspetive 
words in the sentences are lemmatize.
:path the path of the xml file
:wn2bn_mapping the dict that contains the wordnet to babelnet mapping
:file the path where write the sentences taken from the xml file
'''
def make_sentences_file_tom(path,wn2bn_mapping,file):
    sentences = dict()
    get_sentence = None
    result = []
    lemmatizer = WordNetLemmatizer()

    for event, elem in tqdm(ET.iterparse(path, events={'start','end'})):
        if event == 'start':
            #take the word to substitute
            if elem.tag == 'answer':
                wn_offset = elem.attrib["senseId"][3:]
            if elem.tag == 'context':
                sentence = elem.text

            if elem.tag == 'head' and type(elem.text) == str and type(elem.tail) == str and type(sentence) == str:
                anchor = elem.text
                sentence+= anchor + elem.tail
                get_sentence = True

        if event == 'end':
            if elem.tag == 'instance' and get_sentence:
                if not sentences.get(sentence):
                    sentences[sentence] = set()
                if wn2bn_mapping.get(wn_offset):
                    sentences[sentence].add(anchor+
                                            "|"+'_'.join(lemmatizer.lemmatize(anchor,pos=wn_offset[-1]).lower().split())+"_"+wn2bn_mapping.get(wn_offset))  
                get_sentence = False
        elem.clear()
        
    #process the sentences
    for s in tqdm(sentences.keys()):
        for l in sentences[s]:
            words = l.split("|")
            s = s.replace(words[0],re.search(r"[A-Z|a-z][a-z]*_?[A-Z|a-z]*[a-z]*\_bn\:[0-9]{8}[a|v|n|r]",words[1]).group(0))
        result.append(s)
    
    #write the result
    with open(file,'w',encoding='utf-8') as f:
        for s in result:
            f.write(s+"\n")
        del result    
        f.close()

def main():
    #CREATE THE WORDMAPPING DICT
    print("Making bn2wn_mapping and wn2_bn_mapping...")
    bn_wn_mapping, wn_bn_mapping = make_dict_mapping(path_mapping)
    #CREATE THE TRAIN_X FILE FOR EUROSENSE HIGH_PRECISION AND FOR TOM
    print("Parsing the eurosense high precision xml file...")
    make_sentences_file_eurosense(path_precision,bn_wn_mapping,path_precision_result)
    print("Done!")
    print("Parsing the Tom xml file...")
    make_sentences_file_tom(path_tom,wn_bn_mapping,path_tom_result)
    print("Done!")

if __name__ == "__main__":
    main()