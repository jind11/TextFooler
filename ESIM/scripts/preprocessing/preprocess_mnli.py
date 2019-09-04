"""
Preprocess the MultiNLI dataset and word embeddings to be used by the
ESIM model.
"""
# Aurelien Coet, 2019.

import os
import pickle
import fnmatch
import json

from esim.data import Preprocessor


def preprocess_MNLI_data(inputdir,
                         embeddings_file,
                         targetdir,
                         lowercase=False,
                         ignore_punctuation=False,
                         num_words=None,
                         stopwords=[],
                         labeldict={},
                         bos=None,
                         eos=None):
    """
    Preprocess the data from the MultiNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    matched_dev_file = ""
    mismatched_dev_file = ""
    matched_test_file = ""
    mismatched_test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, '*_train.txt'):
            train_file = file
        elif fnmatch.fnmatch(file, '*_dev_matched.txt'):
            matched_dev_file = file
        elif fnmatch.fnmatch(file, '*_dev_mismatched.txt'):
            mismatched_dev_file = file
        elif fnmatch.fnmatch(file, '*_test_matched_unlabeled.txt'):
            matched_test_file = file
        elif fnmatch.fnmatch(file, '*_test_mismatched_unlabeled.txt'):
            mismatched_test_file = file

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                stopwords=stopwords,
                                labeldict=labeldict,
                                bos=bos,
                                eos=eos)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "worddict.pkl"), 'wb') as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev sets ", 20*"=")
    print("\t* Reading matched dev data...")
    data = preprocessor.read_data(os.path.join(inputdir, matched_dev_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "matched_dev_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print("\t* Reading mismatched dev data...")
    data = preprocessor.read_data(os.path.join(inputdir, mismatched_dev_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "mismatched_dev_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # # -------------------- Test data preprocessing -------------------- #
    # print(20*"=", " Preprocessing test sets ", 20*"=")
    # print("\t* Reading matched test data...")
    # data = preprocessor.read_data(os.path.join(inputdir, matched_test_file))
    #
    # print("\t* Transforming words in premises and hypotheses to indices...")
    # transformed_data = preprocessor.transform_to_indices(data)
    # print("\t* Saving result...")
    # with open(os.path.join(targetdir, "matched_test_data.pkl"), 'wb') as pkl_file:
    #     pickle.dump(transformed_data, pkl_file)
    #
    # print("\t* Reading mismatched test data...")
    # data = preprocessor.read_data(os.path.join(inputdir, mismatched_test_file))
    #
    # print("\t* Transforming words in premises and hypotheses to indices...")
    # transformed_data = preprocessor.transform_to_indices(data)
    # print("\t* Saving result...")
    # with open(os.path.join(targetdir, "mismatched_test_data.pkl"), 'wb') as pkl_file:
    #     pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), 'wb') as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess the MultiNLI dataset')
    parser.add_argument('--config',
                        default="../config/preprocessing/mnli_preprocessing.json",
                        help='Path to a configuration file for preprocessing MultiNLI')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as cfg_file:
        config = json.load(cfg_file)

    preprocess_MNLI_data(os.path.normpath(config["data_dir"]),
                         os.path.normpath(config["embeddings_file"]),
                         os.path.normpath(config["target_dir"]),
                         lowercase=config["lowercase"],
                         ignore_punctuation=config["ignore_punctuation"],
                         num_words=config["num_words"],
                         stopwords=config["stopwords"],
                         labeldict=config["labeldict"],
                         bos=config["bos"],
                         eos=config["eos"])
