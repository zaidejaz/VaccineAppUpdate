import pandas as pd
import numpy as np
from wordcloud import WordCloud
import seaborn as sns
from gensim.models.coherencemodel import CoherenceModel

import gensim
from gensim.models import CoherenceModel


#Prepare objects for LSA gensim implementation
import gensim.corpora as corpora
#Running LSA
from gensim.models.lsimodel import LsiModel


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import ast

class LSA():
    def __init__(self, data_path):
        
        data = pd.read_csv(data_path, usecols=['Stem_Tweets', 'Stem_String'])
        data = data.dropna()
        self.data = data


    def process_tokens(self, token_list):
  
        try:

            tokens = ast.literal_eval(token_list )
            if len(tokens) ==0:
                return np.nan
    
            return ' '.join(tokens)
  
        except:
            return np.nan

    def get_wordcloud(self, column, saved_path):

        b = self.data[column].tolist()
        b = ' '.join(map(str, b))

        wordcloud = WordCloud(max_font_size=40, max_words=1000, background_color="white", random_state=100,
                          prefer_horizontal=0.60).generate(b.lower())
        plt.figure(figsize=(12,8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(saved_path)
        plt.show()
        

    def get_corpus(self, column_name ):

        #decomposing sentences into tokens
        tokens = [sentence.split(' ') for sentence in self.data[ column_name ] ]
        # training a bi gram model in order to include those bigrams as tokens who occured at least 6 times
        # in the whole dataset
        bigram = gensim.models.Phrases(tokens, min_count=2, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        # including bigrams as tokens 
        sents = [ bigram_mod[token] for token in tokens]

        # Create Dictionary to keep track of vocab
        dct = corpora.Dictionary(tokens)

        print('Unique words before filtering/after pre-processing', len(dct))
        # no_below= 3
        # filter the words that occure in less than 3 documents and in more the 60% of documents
        dct.filter_extremes(no_below= 3, no_above=0.60 )
        print('Unique words after filtering', len(dct))

        # Create Corpus
        corpus = [dct.doc2bow(sent) for sent in sents]

        tfidf = gensim.models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        return corpus_tfidf, dct, sents


    def get_coherence_graph(self, scores):

        plt.figure(figsize=(10, 5))
        plt.plot(list(range(2, 8)), scores, marker='o', color='green')
        sns.despine(top=True, right=True, left=False, bottom=False)

        plt.locator_params(integer=True)
        plt.title('Coherence score vs the number of topics for LSA')
        plt.xlabel('Number of topics')
        plt.ylabel('Coherence Scores')
        plt.savefig('lsa_scores.jpg')
        plt.show()

    def get_best_number_of_topics(self, sents, corpus_tfidf, dct):

        scores = []
        start=2
        for k in range(start, 8):
            # Lsa model
            lsa_model = gensim.models.LsiModel(  corpus=corpus_tfidf, num_topics=k,
                                                        id2word=dct)
            # to calculate score for coherence
            coherence_model_lsa = CoherenceModel(model=lsa_model, texts=sents, dictionary=dct, coherence='c_v')
            coherence_lsa = coherence_model_lsa.get_coherence()
            # print(k, coherence_lsa)
            scores.append(coherence_lsa)

        self.get_coherence_graph(scores)
        best_topicno = np.argmax(scores)+start
        best_score = scores[best_topicno-start]
        return best_score, best_topicno

    def get_wordclouds_topics(self, lsa_model):
        
        for t in range(lsa_model.num_topics):
            try:  
              plt.figure(figsize=(8,5))
              plt.imshow(WordCloud().fit_words( dict( lsa_model.show_topic(t, 200)) ) )
              plt.axis("off")
              plt.title("Topic #" + str(t))
              plt.savefig(f'Topic # {t}')
              plt.show()
            except:
              print('Not enough words for a topic')


    def best_model(self, nmb_of_topics, corpus_tfidf, dct):
        
        lsa_model = LsiModel(  corpus=corpus_tfidf, num_topics=nmb_of_topics, id2word=dct)
        
        print('*****Words Under each topic********\n')
        for pair in lsa_model.print_topics():
          print(f'Topic {pair[0]}: ', pair[1])
        self.get_wordclouds_topics(lsa_model)

        return lsa_model


    def get_lsa_topics(self):
        
        self.data['processed'] = self.data['Stem_Tweets'].apply(self.process_tokens)
        self.data.dropna(subset=['processed'], inplace=True)
        
        # get the word cloud about the frequency of words
        self.get_wordcloud('processed', 'word_cloud.jpg')

        # get the corpus
        corpus_tfidf, dct, sents = self.get_corpus( 'processed')

        # train the LSA model
        best_score, best_nmb_of_topics = self.get_best_number_of_topics(sents, corpus_tfidf, dct)

        print(f'The selected number of topics are: {best_nmb_of_topics} with a score {best_score}' )

        self.best_model( best_nmb_of_topics, corpus_tfidf, dct)


if __name__=='__main__':

    data_path = './DataSetStemmed.csv'
    print(data_path)
    lsa = LSA(data_path)
    lsa.get_lsa_topics()






    