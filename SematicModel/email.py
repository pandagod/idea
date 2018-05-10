import mysql.connector
import random
import pickle
import os
import en_core_web_sm
import nlp_prepare as util
from langdetect import detect
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

train_dir = os.path.join(os.path.curdir, './email')
data_dir = os.path.join(train_dir, 'data')

for dir in [train_dir, data_dir]:
  if not os.path.exists(dir):
    os.makedirs(dir)

trainset_fn = os.path.join(data_dir, 'train.dataset')
devset_fn = os.path.join(data_dir, 'dev.dataset')
testset_fn = os.path.join(data_dir, 'test.dataset')
ngrams_fn = os.path.join(data_dir, 'ngrams.pickle')
ngrams_length_fn = os.path.join(data_dir, 'ngrams_length.pickle')

def build_ngrams_model():
    try:
        with open(ngrams_fn, 'rb') as vocab_file:
            model = pickle.load(vocab_file)
            print('ngrams model loaded')
            return model
    except IOError:
        print('building ngrams model')


    nlp = en_core_web_sm.load()

    Corpus = []
    results = load_data_from_db()
    for row in results:
        inbound_body = util.clean_str(row[1] + " " + row[2])
        outbound_body = util.clean_str(row[3] + " " + row[4])
        try:
            if inbound_body != '' and detect(inbound_body) == 'en':
                inbound_body = inbound_body.decode('utf-8', 'ignore')
                for sent in nlp(inbound_body).sents:
                    Corpus.append(sent.text)
            if outbound_body != '' and detect(outbound_body) == 'en':
                outbound_body = outbound_body.decode('utf-8', 'ignore')
                for sent in nlp(outbound_body).sents:
                    Corpus.append(sent.text)

        except Exception, e:
            print 'str(Exception):\t', str(e)
            print(row[0])

    model = CountVectorizer(tokenizer=util.CustomizeTokenizer(nlp), ngram_range=(3, 3), min_df=5,
                           stop_words=['a', 'an', 'the', 'of', '-PRON-', 're', 'regards', 'thank', 'helvetica', 'neue'])
    model.fit_transform(Corpus)

    with open(ngrams_fn, 'wb') as ngrams_file:
        pickle.dump(ngrams_file, ngrams_fn)

    return model

def load_data_from_db():
    cnx = mysql.connector.connect(host="10.249.71.213", user="root", password="root", database="ai")
    cursor = cnx.cursor()

    sql = ("select sr_number,inbound_subject_0,inbound_body_0,outbound_subject_0,outbound_body_0 "
           "from ai.2017_full_inbound_outbound where inbound_subject_0!='' and inbound_body_0!='' and outbound_subject_0!='' and inbound_body_0!='' "
           "limit 50;")

    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        print results
    except Exception, e:
        print 'str(e):\t\t', str(e)

    cursor.close()
    cnx.close()
    return results

def convertSparseToIds(vector):
    return [index for index, i in enumerate(vector[0]) if i >0]

def load_data(ngrams_model):
    DataSet = []
    results =load_data_from_db

    for row in results:
        inbound_body = util.clean_str(row[1] + " " + row[2])
        outbound_body = util.clean_str(row[3] + " " + row[4])
        try:
            if inbound_body != '' and detect(inbound_body) == 'en' and outbound_body != '' and detect(
                    outbound_body) == 'en':
                x = convertSparseToIds(ngrams_model.transform([util.clean_str(inbound_body)]).toarray())
                y = convertSparseToIds(ngrams_model.transform([util.clean_str(outbound_body)]).toarray())
                DataSet.append((x, y))

        except Exception, e:
            print 'str(Exception):\t', str(e)
            print(row[0])

    return DataSet

def ngrams_size():
  ngrmas_len_fn = open(ngrams_length_fn, 'rb')
  ngrmas_size = pickle.load(ngrmas_len_fn)
  return ngrmas_size


def make_data(split_points=(0.9, 0.95)):
    train_ratio, dev_ratio = split_points
    ngrams_model = build_ngrams_model()
    ngrams_len_fn = open(ngrams_length_fn, 'wb')
    pickle.dump(len(ngrams_model.vocabulary_), ngrams_len_fn)
    ngrams_len_fn.close()
    train_f = open('trainset_fn', 'wb')
    dev_f = open('devset_fn', 'wb')
    test_f = open('testset_fn', 'wb')

    try:
        DataSet = list(load_data(ngrams_model))
        random.shuffle(DataSet)
        for piece in tqdm(DataSet):
            r = random.random()
            if r < train_ratio:
                f = train_f
            elif r < dev_ratio:
                f = dev_f
            else:
                f = test_f
            pickle.dump(piece, f)

    except KeyboardInterrupt:
        pass

    train_f.close()
    dev_f.close()
    test_f.close()


def _read_dataset(fn, epochs=1):
    c = 0
    while 1:
        c += 1
        if epochs > 0 and c > epochs:
            return
        print('epoch %s' % c)
        with open(fn, 'rb') as f:
            try:
                while 1:
                    x, y = pickle.load(f)
                    yield x, y
            except EOFError:
                continue


def read_trainset(epochs=1):
    return _read_dataset('trainset_fn', epochs=epochs)


def read_devset(epochs=1):
    return _read_dataset('devset_fn', epochs=epochs)


def read_testset(epochs=1):
    return _read_dataset('testset_fn', epochs=epochs)

if __name__ == '__main__':
    make_data()