"""
Test the ESIM model on the preprocessed MultiNLI dataset.
"""
# Aurelien Coet, 2019.

import os
import pickle
import argparse
import torch
import json

from torch.utils.data import DataLoader
from esim.data import NLIDataset
from esim.model import ESIM


def predict(model, dataloader, labeldict):
    """
    Predict the labels of an unlabelled test set with a pretrained model.

    Args:
        model: The torch module which must be used to make predictions.
        dataloader: A DataLoader object to iterate over some dataset.
        labeldict: A dictionary associating labels to integer values.

    Returns:
        A dictionary associating pair ids to predicted labels.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    # Revert the labeldict to associate integers to labels.
    labels = {index: label for label, index in labeldict.items()}
    predictions = {}

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:

            # Move input and output data to the GPU if one is used.
            ids = batch["id"]
            premises = batch['premise'].to(device)
            premises_lengths = batch['premise_length'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypotheses_lengths = batch['hypothesis_length'].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            _, preds = probs.max(dim=1)

            for i, pair_id in enumerate(ids):
                predictions[pair_id] = labels[int(preds[i])]

    return predictions


def main(test_files, pretrained_file, labeldict, output_dir, batch_size=32):
    """
    Test the ESIM model with pretrained weights on the MultiNLI dataset.

    Args:
        test_files: The paths to the preprocessed matched and mismatched MNLI
            test sets.
        pretrained_file: The path to a checkpoint produced by the
            'train_mnli' script.
        labeldict: A dictionary associating labels (classes) to integer values.
        output_dir: The path to a directory where the predictions of the model
            must be saved.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    output_dir = os.path.normpath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = torch.load(pretrained_file)

    # Retrieve model parameters from the checkpoint.
    vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
    embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
    hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
    num_classes = checkpoint['model']['_classification.4.weight'].size(0)

    print("\t* Loading test data...")
    with open(os.path.normpath(test_files["matched"]), 'rb') as pkl:
        matched_test_data = NLIDataset(pickle.load(pkl))
    with open(os.path.normpath(test_files["mismatched"]), 'rb') as pkl:
        mismatched_test_data = NLIDataset(pickle.load(pkl))

    matched_test_loader = DataLoader(matched_test_data,
                                     shuffle=False,
                                     batch_size=batch_size)
    mismatched_test_loader = DataLoader(mismatched_test_data,
                                        shuffle=False,
                                        batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint['model'])

    print(20 * "=",
          " Prediction on MNLI with ESIM model on device: {} ".format(device),
          20 * "=")

    print("\t* Prediction for matched test set...")
    predictions = predict(model, matched_test_loader, labeldict)

    with open(os.path.join(output_dir, "matched_predictions.csv"), 'w') as output_f:
        output_f.write("pairID,gold_label\n")
        for pair_id in predictions:
            output_f.write(pair_id+","+predictions[pair_id]+"\n")

    print("\t* Prediction for mismatched test set...")
    predictions = predict(model, mismatched_test_loader, labeldict)

    with open(os.path.join(output_dir, "mismatched_predictions.csv"), 'w') as output_f:
        output_f.write("pairID,gold_label\n")
        for pair_id in predictions:
            output_f.write(pair_id+","+predictions[pair_id]+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 the MNLI matched and mismatched test sets')
    parser.add_argument('checkpoint',
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument('--config', default='../config/testing/mnli_testing.json',
                        help='Path to a configuration file')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as config_file:
        config = json.load(config_file)

    main(config['test_files'],
         args.checkpoint,
         config['labeldict'],
         config['output_dir'],
         config['batch_size'])
