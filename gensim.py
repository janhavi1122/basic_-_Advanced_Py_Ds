# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:38:10 2023

@author: santo
"""

import gensim
import pandas as pd
df = pd.read_json("C:/datasets/reviews_Cell_Phones_and_Accessories_5.json.gz",lines=True)
df
df.shape
#simple Preprocessing and Tokenization
review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
review_text
review_text.loc[0]
df.reviewText.loc[0]
model=gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4,
)
''' where window is how many words u are going to consider as sliding window u can choose any count min_count tere must min 2 words in each sentence workers.no.of threads
'''
 #build vocabulary
model.build_vocab(review_text,progress_per=1000)
#progress_pre after 1000 words it shows progress
#Train the Word2Vec model
# it will take time
model.train(review_text,total_examples = model.corpus_count,epochs=model.epochs)
#save model
model.save("C:/py1./word2vec-amazon-cell-accessories=reviews-short.model")
#finding Similar word 
model.wv.similarity("bad")
model.wv.similarity(w1="cheap", w2="inexpensive")
model.wv.similarity(w1="great", w2="good")