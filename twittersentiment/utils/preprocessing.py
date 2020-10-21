from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torchtext.vocab import GloVe
import numpy as np
import torch
import pickle
import os
import nltk
import logging


logging.basicConfig(format="%(message)s", level=logging.INFO)
MAX_LEN = 80


class Preprocess:
    @staticmethod
    def preprocess_text(text: list):
        """Function to preprocess and create corpus"""
        new_corpus = []
        for text in text:
            words = [w for w in text.split()]

            new_corpus.append(words)
        return new_corpus

    @staticmethod
    def tokenizer(corpus, path, mode="train"):

        """
        corpus : the corpus to tokenize and pad
        path : tokenizer path
        mode : train/test

        """

        if not path.endswith(".pickle"):
            path = os.path.join(path, "tokenizer.pickle")

        if mode == "train":
            tokenizer_obj = Tokenizer()
            tokenizer_obj.fit_on_texts(corpus)
            word_index = tokenizer_obj.word_index

            if not os.path.exists(path):
                os.mkdir(path)

            with open(path, "wb") as tok:
                pickle.dump(tokenizer_obj, tok, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            word_index = None
            with open(path, "rb") as tok:
                tokenizer_obj = pickle.load(tok)

        sequences = tokenizer_obj.texts_to_sequences(corpus)
        tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating="post", padding="post")

        return tweet_pad, word_index, path

    def prepare_matrix(word_index, name, dim):

        """
        word_index : tokenizer word index
        name : name of embedding vector
        available vectors

                charngram.100d
                fasttext.en.300d
                fasttext.simple.300d
                glove.42B.300d
                glove.840B.300d
                glove.twitter.27B.25d
                glove.twitter.27B.50d
                glove.twitter.27B.100d
                glove.twitter.27B.200d
                glove.6B.50d
                glove.6B.100d
                glove.6B.200d
                glove.6B.300d


        dim : embedding dimension

        """
        logging.info("Preparing vectors, this might take a bit..")
        embedding_dict = GloVe(name, dim=dim)
        num_words = len(word_index)
        embedding_matrix = np.zeros((num_words + 1, dim))

        for word, i in word_index.items():
            if i > num_words:
                continue

            emb_vec = embedding_dict[word]
            if not torch.equal(emb_vec, torch.zeros((dim), dtype=torch.float)):
                embedding_matrix[i] = emb_vec

            elif torch.equal(embedding_dict[word.lower()], torch.zeros((dim), dtype=torch.float)):
                emb_vec = embedding_dict[word.lower()]
                embedding_matrix[i] = emb_vec

            elif torch.equal(embedding_dict[word.title()], torch.zeros((dim), dtype=torch.float)):
                emb_vec = embedding_dict[word.title()]
                embedding_matrix[i] = emb_vec

            else:
                embedding_matrix[i] = embedding_dict[word]

        return embedding_matrix
