! pip install nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
import re
import math
import operator



def tfisf_evaluator(text,length):

    raw_sents = nltk.tokenize.sent_tokenize(text)

    def clean_sent(sent):
        filtered_sent = re.sub('[\W]', ' ', sent) 
        stripped_sent = re.sub(' +', ' ', filtered_sent)
        stripped_sent = re.sub(r"\b[a-zA-Z]\b", "", stripped_sent.strip())
        cleaned_sent = stripped_sent.lower()  
        return cleaned_sent
    
    def strip_sent(sent):
        stripped_sent = re.sub('\s', ' ', sent)
        stripped_sent = re.sub(' +', ' ', stripped_sent.strip())
        return stripped_sent

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_sent(sent):
        lemmatized_sent = []
        for w in nltk.tokenize.word_tokenize(sent):
            if w not in stopwords:
                lemmatized_sent.append(lemmatizer.lemmatize(w, get_wordnet_pos(w)))
        return ' '.join(lemmatized_sent) 

    def get_tfisf_dic(words,sents):
    
        word_freq_dic = {}
    
        for w in words:
            if w not in word_freq_dic.keys():
                word_freq_dic[w] = 1 
            else:
                word_freq_dic[w] += 1 
            
        for item in word_freq_dic:
            word_freq_dic[item] = math.log(word_freq_dic[item] / len(word_freq_dic))
    
        word_rarity_dic = {}
    
        for w in words:
            for sent in sents:
                if w in sent:
                    if w not in word_rarity_dic.keys():
                        word_rarity_dic[w] = 1
                    else:
                        word_rarity_dic[w] += 1
            
        for item in word_rarity_dic:
            word_rarity_dic[item] = math.log(len(sents) / word_rarity_dic[item])
    
        tsisf_dic = word_freq_dic
    
        for item in tsisf_dic:
            tsisf_dic[item] = word_freq_dic[item] * word_rarity_dic[item]
        return tsisf_dic
    
    def get_score_dic(sents,dic):
        score_dic = {}

        for sent in sents:
            for w in nltk.word_tokenize(sent):
                if sent not in score_dic.keys():
                    score_dic[sent] = dic[w]
                else:
                    score_dic[sent] += dic[w]
        return score_dic

    cleaned_sents = []
    for sent in raw_sents:
        cleaned_sents.append(clean_sent(sent))
    lemmatized_sents = []
    for sent in cleaned_sents:
        lemmatized_sents.append(lemmatize_sent(sent))

    lemmatized_words = []
    for sent in lemmatized_sents:
        lemmatized_words = lemmatized_words + nltk.word_tokenize(sent)

    tfisf_dic = get_tfisf_dic(lemmatized_words,lemmatized_sents)

    score_dic = get_score_dic(lemmatized_sents,tfisf_dic)

    output_sents = []
    for sent in raw_sents:
        output_sents.append(strip_sent(sent))

    output_dic = {} 
    for sent in output_sents:
        output_dic[sent] = score_dic[list(score_dic)[output_sents.index(sent)]]

    output_dic = sorted(output_dic.items(), key=operator.itemgetter(1), reverse=True)
    
    return output_dic[:length]
