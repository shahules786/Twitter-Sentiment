from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torchtext.vocab import GloVe
import numpy as np
import torch
import pickle
from tqdm import tqdm
import os
import nltk
from nltk.tokenize import word_tokenize

nltk.download("all")

MAX_LEN = 80


class Preprocess:
    @staticmethod
    def preprocess_text(text: list):
        """Function to preprocess and create corpus"""
        new_corpus = []
        for text in tqdm(text):
            words = [w for w in word_tokenize(text)]

            new_corpus.append(words)
        return new_corpus

    @staticmethod
    def tokenizer(corpus, path, mode="train"):

        if mode == "train":
            path = os.path.join("tokenizer", "new_tokenizer.pickle")
            tokenizer_obj = Tokenizer()
            tokenizer_obj.fit_on_texts(corpus)
            word_index = tokenizer_obj.word_index

            with open(path, "wb") as tok:
                pickle.dump(tokenizer_obj, tok, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            word_index = None
            with open(path, "rb") as tok:
                tokenizer = pickle.load(tok)

        sequences = tokenizer.texts_to_sequences(corpus)
        tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating="post", padding="post")

        return tweet_pad, word_index, path

    def prepare_matrix(word_index):
        embedding_dict = GloVe("twitter.27B", dim=200)

        num_words = len(word_index)
        embedding_matrix = np.zeros((num_words + 1, 200))

        for word, i in tqdm(word_index.items()):
            if i > num_words:
                continue

            emb_vec = embedding_dict[word]
            if not torch.equal(emb_vec, torch.zeros((200), dtype=torch.float)):
                embedding_matrix[i] = emb_vec

            elif torch.equal(embedding_dict[word.lower()], torch.zeros((200), dtype=torch.float)):
                emb_vec = embedding_dict[word.lower()]
                embedding_matrix[i] = emb_vec

            elif torch.equal(embedding_dict[word.title()], torch.zeros((200), dtype=torch.float)):
                emb_vec = embedding_dict[word.title()]
                embedding_matrix[i] = emb_vec

            else:
                embedding_matrix[i] = embedding_dict[word]

        return embedding_matrix
