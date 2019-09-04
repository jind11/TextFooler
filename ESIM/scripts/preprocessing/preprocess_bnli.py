"""
Preprocess the Breaking NLI data set.
"""

import os
import json
import pickle

from nltk import word_tokenize

from esim.data import Preprocessor


def jsonl_to_txt(input_file, output_file):
    """
    Transform the Breaking NLI data from a jsonl file to .txt for
    further processing.

    Args:
        input_file: The path to the Breaking NLI data set in jsonl format.
        output_file: The path to the .txt file where the tranformed data must
            be saved.
    """
    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        output_f.write("label\tsentence1\tsentence2\t\t\t\t\t\tpairID\n")

        for line in input_f:
            data = json.loads(line)

            # Sentences in the Breaking NLI data set aren't distributed in the
            # form of binary parses, so we must tokenise them with nltk.
            sentence1 = word_tokenize(data['sentence1'])
            sentence1 = " ".join(sentence1)
            sentence2 = word_tokenize(data['sentence2'])
            sentence2 = " ".join(sentence2)

            # The 5 tabs between sentence 2 and the pairID are added to
            # follow the same structure as the txt files in SNLI and MNLI.
            output_f.write(data['gold_label'] + "\t" + sentence1 + "\t" +
                           sentence2 + "\t\t\t\t\t" + str(data['pairID']) +
                           "\n")


def preprocess_BNLI_data(input_file,
                         targetdir,
                         worddict,
                         labeldict):
    """
    Preprocess the BNLI data set so it can be used to test a model trained
    on SNLI.

    Args:
        inputdir: The path to the file containing the Breaking NLI (BNLI) data.
        target_dir: The path to the directory where the preprocessed Breaking
            NLI data must be saved.
        worddict: The path to the pickled worddict used for preprocessing the
            training data on which models were trained before being tested on
            BNLI.
        labeldict: The dict of labels used for the training data on which
            models were trained before being tested on BNLI.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    output_file = os.path.join(targetdir, "bnli.txt")

    print(20*"=", " Preprocessing Breaking NLI data set ", 20*"=")
    print("\t* Tranforming jsonl data to txt...")
    jsonl_to_txt(input_file, output_file)

    preprocessor = Preprocessor(labeldict=labeldict)

    with open(worddict, 'rb') as pkl:
        wdict = pickle.load(pkl)
    preprocessor.worddict = wdict

    print("\t* Reading txt data...")
    data = preprocessor.read_data(output_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "bnli_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess the Breaking\
 NLI (BNLI) dataset')
    parser.add_argument('--config',
                        default="../config/preprocessing/bnli_preprocessing.json",
                        help='Path to a configuration file for preprocessing BNLI')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as cfg_file:
        config = json.load(cfg_file)

    preprocess_BNLI_data(os.path.normpath(config["data_file"]),
                         os.path.normpath(config["target_dir"]),
                         os.path.normpath(config["worddict"]),
                         config["labeldict"])
