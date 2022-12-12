
# Import Pandas
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.backend import clear_session
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, Dense
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
import numpy as np
import pandas as pd
# Settingan di Pandas untuk menghilangkan warning
pd.options.mode.chained_assignment = None  # default='warn'

# Import Numpy

# Loading Bar TQDM

# Stopword Removal
nltk.download('stopwords')

# Grafik
sns.set_style("whitegrid")
np.random.seed(0)

testSize = 0.15  # Pembagian ukuran datatesing

MAX_NB_WORDS = 100000  # Maximum jumlah kata pada vocabulary yang akan dibuat
max_seq_len = 46  # Panjang kalimat maximum

num_epochs = 40  # Jumlah perulangan / epoch saat proses training

aspek_kategori = ['akurasi', 'kualitas',
                  'pelayanan', 'pengemasan', 'harga', 'pengiriman']


def read_data(filename):
    # Membaca file CSV ke dalam Dataframe
    df = pd.read_csv(filename)

    # Encoding Categorical Data (Mengubah data kategorikal menjadi angka)
    label_dict = {'negatif': 0, 'positif': 1, '-': 99}
    df['s_akurasi'] = df['s_akurasi'].replace(label_dict)
    df['s_kualitas'] = df['s_kualitas'].replace(label_dict)
    df['s_pelayanan'] = df['s_pelayanan'].replace(label_dict)
    df['s_pengemasan'] = df['s_pengemasan'].replace(label_dict)
    df['s_harga'] = df['s_harga'].replace(label_dict)
    df['s_pengiriman'] = df['s_pengiriman'].replace(label_dict)

    # Membagi dataframe menjadi data training & testing
    df_training, df_testing = train_test_split(
        df, test_size=testSize, random_state=42, shuffle=True)

    # Reset Index
    df_training = df_training.reset_index()
    df_testing = df_testing.reset_index()

    return df_training, df_testing


def preprocessing(data):
    # Case Folding
    data['lower'] = data['teks'].str.lower()

    # Punctual Removal
    data['punctual'] = data['lower'].str.replace('[^a-zA-Z]+', ' ', regex=True)

    # Normalization
    kamus_baku = pd.read_csv('kata_baku.csv', sep=";")
    dict_kamus_baku = kamus_baku[['slang', 'baku']].to_dict('list')
    dict_kamus_baku = dict(
        zip(dict_kamus_baku['slang'], dict_kamus_baku['baku']))
    norm = []
    for i in data['punctual']:
        res = " ".join(dict_kamus_baku.get(x, x) for x in str(i).split())
        norm.append(str(res))
    data['normalize'] = norm

    # Stopword Removal
    stop_words = set(stopwords.words('indonesian'))
    swr = []
    for i in tqdm(data['normalize']):
        tokens = word_tokenize(i)
        filtered = [word for word in tokens if word not in stop_words]
        swr.append(" ".join(filtered))
    data['stopwords'] = swr

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem = []
    for i in tqdm(data['stopwords']):
        stem.append(stemmer.stem(str(i)))
    data['stemmed'] = stem

    return data

# Tokenisasi Training


def tokenize_training(data_training):
    # Menggunakan variabel global agar 'tokenizer' bisa dipake di luar fungsi ini
    global tokenizer
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=False)
    tokenizer.fit_on_texts(data_training['stemmed'])
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(data_training['stemmed'])
    word_seq_train = pad_sequences(train_sequences, maxlen=max_seq_len)

    return word_index, word_seq_train

# Tokenisasi Testing


def tokenize_testing(data_testing):
    test_sequences = tokenizer.texts_to_sequences(data_testing['stemmed'])
    word_seq_test = pad_sequences(test_sequences, maxlen=max_seq_len)

    return word_seq_test

# Tokenisasi secara umum


def tokenize(data):
    sequences = tokenizer.texts_to_sequences(data['stemmed'])
    word_seq = pad_sequences(sequences, maxlen=max_seq_len)

    return word_seq

# Pendefenisian fungsi word embedding


def word_embedding(word_index):
    fasttext_word_to_index = pickle.load(
        open("../Pertemuan 6 - Model Building & Training/fasttext_voc", 'rb'))

    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)

    embed_dim = 300  # dimensi matrix (dari fastTextnya cc.id.300.vec)
    embedding_matrix = np.zeros((nb_words, embed_dim))

    for word, index in word_index.items():
        if index < nb_words:
            embedding_vector = fasttext_word_to_index.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                embedding_matrix[index] = embedding_vector
            else:
                words_not_found.append(word)

    return embedding_matrix, nb_words, embed_dim
