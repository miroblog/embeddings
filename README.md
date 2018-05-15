
### First : pip install -r requirements.txt

***

### Second : make embeddings and do sentiment analysis

|          | skipgram | cbow | glove | fastText(syll) | fastText(jamo) |
|:--------:|:--------:|:----:|:-----:|:--------------:|:--------------:|
|   word   |    (1)   |  (2) |  (3)  |       (4)      |       (5)      |
| morpheme |    (6)   |  (7) |  (8)  |       (9)      |      (10)      |


<pre><code>
    parser.add_argument("-p", "--parser", default="twitter", help="morpheme parser: kkma, komoran, twitter")
    parser.add_argument("-m", "--model", help="model name : word2vec, fastText, glove")
    parser.add_argument("-u", "--unit", help="unit : WORD, MORPHEME, SYLLABLE, JAMO")
    parser.add_argument("-o", "--output_path", help="output path")
    parser.add_argument("-v", "--variation", default='sg', help="variation : sg, cbow")
    parser.add_argument("-min", "--min_ngram", default=1, help="minimum n gram: 1~3")
</pre></code>



1. python nsmc_args.py -m word2vec -u WORD -o history -v sg
2. python nsmc_args.py -m word2vec -u WORD -o history -v cbow
3. python nsmc_args.py -m glove -u WORD -o history -v cbow
4. python nsmc_args.py -m fastText -u SYLLABLE -o history -v sg
5. python nsmc_args.py -m fastText -u JAMO -o history -v sg 
6. python nsmc_args.py -p twitter -m word2vec -u MORPHEME -o history -v sg
7. python nsmc_args.py -p twitter -m word2vec -u MORPHEME -o history -v cbow
8. python nsmc_args.py -p twitter -m glove -u MORPHEME -o history
9. python nsmc_args.py -p twitter -m fastText -u SYLLABLE+MORPHEME -o history -v sg
10. python nsmc_args.py -p twitter -m fastText -u JAMO+MORPHEME -o history -v sg

#### minimum n gram: python nsmc_args.py -p twitter -m fastText -u JAMO+MORPHEME -o history -v sg -min 1

***

### Third: extract history, (dumped history files to a single .csv file)

1. -p : path for history
2. -o : output file name

#### python history_extractor.py -p ./entire_history -o history

***

### Fourth : visualize result
1. https://github.com/miroblog/embeddings/blob/master/notebook_visualize.ipynb

***

### PATH 

1. ./training_samples/ : data x, label y for the lstm
2. ./preprocessed/ : processed texts and tokens
3. ./embeddings/ : embeddings in word2vec text format, used for making an embedding layer
4. ./entire_history/ : dumped history
5. ./nsmc/ : training corpus, naver movie review

***

### Results are in : ./result_csv/

  