# coding:utf-8

import re
import gensim
import itertools
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import sys,os
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import datetime
import pickle
reload(sys)
sys.setdefaultencoding('utf8')




movieFile=  'movies.dat'
ratingFile = 'ratings.dat'
userFile = 'users.dat'



def remove_illegal(temp):
    temp = str(temp).lower()
    string = re.sub("[0-9\s+\.\!\/_\-\:,$%^*()`&#<>+\"\']",' ',temp)
    return string

if __name__ == "__main__":

                                    
    movie = open(movieFile,'rb')
    m = []
    for line in movie.readlines():
        linelist = line.strip().split('::')
        if len(linelist) != 3:
            print line
        else:
            m.append(linelist)

    ratings = open(ratingFile,'rb')
    r = []
    for line in ratings.readlines():
        linelist = line.strip().split('::')
        if len(linelist) != 4:
            print line
        else:
            r.append(linelist)

    user = open(userFile,'rb')
    u = []
    for line in user.readlines():
        linelist = line.strip().split('::')
        if len(linelist) != 5:
            print line
        else:
            u.append(linelist)
    

    # word2vec = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary = True)
    movies_df = pd.read_csv('movies.csv',header = 0)
    movies_df['plot'] = movies_df['plot'].apply(remove_illegal)
    Plot = []
    movies_df = movies_df[movies_df['plot'] != ""].reset_index(drop = True) 
    ids = set(movies_df['id'])
    print movies_df
    # for index,line in movies_df.iterrows():
    #     tmp = []
    #     text = line['plot']
    #     text_list = text.strip().split(' ')
    #     for word in text_list:
    #         if word in word2vec.vocab:
    #             tmp.append(word)
    #     Plot.append(tmp)

    # pickle.dump(Plot,open("Plot.pkl",'wb'))


    # plot = pickle.load(open("Plot.pkl",'rb'))
    # li = set(itertools.chain.from_iterable(plot))
    # vocab = {}
    # i = 0
    # for word in li:
    #     vocab[word] = i
    #     i = i+1
    # reverse_vocab = dict(zip(vocab.values(), vocab.keys()))
    # texts = []
    # for li in plot:
    #     texts.append(' '.join(li))
    # vectorize = CountVectorizer(vocabulary = vocab)
    # transform = TfidfTransformer()
    # texts = transform.fit_transform(vectorize.fit_transform(texts)).todense()
    # total = []
    # for i in range(texts.shape[0]):
    #     tmp = 0
    #     a = np.zeros(shape= (300,))
    #     for j in range(texts.shape[1]):
    #         if texts[i,j] > 0:
    #             print texts[i,j]
    #             a = a+ texts[i,j]*word2vec[reverse_vocab[j]]
    #             tmp += texts[i,j]
    #     a = a/tmp
    #     total.append(a)
    # pickle.dump(total,open("plot_embedding.pkl",'wb'))



    plot_embedding = pickle.load(open("plot_embedding.pkl",'rb'))
    genres = []
    for index,line in movies_df.iterrows():
        genre = line['genre'].replace('|',',')
        genres.append(genre)
    movies_df['Language'] = movies_df['Language'].fillna('English')
    movies_df['Country'] = movies_df['Country'].fillna('USA')
    country = list(movies_df['Country'])
    Language = list(movies_df['Language'])
    vectorize = CountVectorizer(min_df = 10)
    genre_vec = vectorize.fit_transform(genres).todense()
    print genre_vec.shape
    country_vec = vectorize.fit_transform(country).todense()
    print country_vec.shape
    language_vec = vectorize.fit_transform(Language).todense()
    print language_vec.shape
    rating = np.array(movies_df['imdbRating']).reshape(3883,1)

    feature = np.hstack((plot_embedding,country_vec,language_vec,genre_vec,rating))
    feature2 = plot_embedding
    feature3 = np.hstack((country_vec,language_vec,genre_vec,rating))
    print feature3.shape
    featureId = list(movies_df['id'])
    print featureId
    featureDict = dict(zip(featureId,list(feature)))
    plotDict = dict(zip(featureId,list(feature2)))
    categoryDict = dict(zip(featureId,list(feature3)))
    # pickle.dump(plotDict,open('feature_plot.pkl','wb'))
    # pickle.dump(categoryDict,open('feature_category.pkl','wb'))
    # pickle.dump(featureDict,open('featureDict.pkl','wb'))



    user = pd.DataFrame(u)
    user.columns = ['uid','sex','age','occup','loc']
    vectorize = CountVectorizer(min_df = 1)
    sex_vec = []
    for i in user['sex']:
        tmp = [1,0] if i == 'F' else [0,1]
        sex_vec.append(tmp)
    sex_vec = np.array(sex_vec)
    age_vec = vectorize.fit_transform(user['age']).todense()
    occup = vectorize.fit_transform(user['occup']).todense()
    feature = np.hstack((age_vec,sex_vec,occup))
    pickle.dump(feature,open('userFeature.pkl','wb'))

    '''
    rating_df = pd.DataFrame(r)
    rating_df.columns = ['uid','Iid','score','time']
    rating_df['time'] = rating_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(float(x)))
    rating_df = rating_df.sort(columns = ['uid','time']).reset_index(drop = True)


    rating_df['Iid'] = rating_df['Iid'].apply(lambda x : int(x))
    groups = rating_df.groupby(['Iid'])

    users = []
    for index,group in groups:
        if group.iloc[0]['Iid'] not in ids:
            continue
        else:
            group = group.sort(columns =['time']).reset_index(drop = True)
            users.append(list(group['uid']))
    users = np.array(users)
    print len(groups)
    print len(ids)
    print users.shape

    items = []
    rating_df['Iid'] = rating_df['Iid'].apply(lambda x : str(x))
    rating_df['uid'] = rating_df['uid'].apply(lambda x : int(x))
    groups = rating_df.groupby(['uid'])
    for index,group in groups:
        group = group.sort(columns = ['time']).reset_index(drop = True)
        items.append(list(group['Iid']))
    items = np.array(items)
    print items.shape


    model = gensim.models.Word2Vec(users,size = 300,workers = 4)
    user2vec =  {}
    for i in range(1,6041):
        try:
            user2vec[i] = model.wv[str(i)]
        except Exception,e:
            pass

    pickle.dump(user2vec,open('user2Vec.pkl','wb'))

    item2vec = {}
    model2 = gensim.models.Word2Vec(items,size = 300,workers = 4)n
    items_vectors = model2.wv
    for item_id in ids:
        try:
            item2vec[item_id] = model2.wv[str(item_id)]
        except Exception,e:
            pass

    pickle.dump(item2vec,open('item2vec.pkl','wb'))
    '''




