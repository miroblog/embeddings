import numpy as np
import pandas as pd
import nltk
from konlpy.tag import Mecab

mecab= Mecab()

tbl = pd.read_csv("D:/naver_movie_data/nsmc/ratings_train.txt", sep='\t')
keywords = [mecab.morphs(str(i).strip()) for i in tbl['document']]
