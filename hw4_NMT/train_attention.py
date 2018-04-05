"""Model training script.

Usage:
    python train.py --config config.yaml
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import numpy as np
import os
import shutil
import torch
import yaml
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable

from model_attention import EncoderRNN, Attn, AttnDecoderRNN
from utils import ShakespeareDataset, Vocab


FLAGS = None
USE_CUDA = torch.cuda.is_available()

def main():
    # Load the configuration file.
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    # Create the checkpoint directory if it does not already exist.
    ckpt_dir = os.path.join(config['data']['ckpt'], config['experiment_name'])
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # Check if a pre-existing configuration file exists and matches the current
    # configuration. Otherwise save a copy of the configuration to the
    # checkpoint directory.
    prev_config_path = os.path.join(ckpt_dir, 'config.yaml')
    if os.path.exists(prev_config_path):
        with open(prev_config_path, 'r') as f:
            prev_config = yaml.load(f)
        assert config == prev_config
    else:
        shutil.copyfile('config.yaml', prev_config_path)

    # Load the vocabularies.
    src_vocab = Vocab.load(config['data']['src']['vocab'])
    tgt_vocab = Vocab.load(config['data']['tgt']['vocab'])

    # Load the training and dev datasets.
    train_data = ShakespeareDataset('train', config, src_vocab, tgt_vocab)
    dev_data = ShakespeareDataset('dev', config, src_vocab, tgt_vocab)

    # Build the model.
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    encoder = EncoderRNN(src_vocab_size, config['model']['embedding_dim'],config['model']['layer'])
    attn = 'general'
    decoder = AttnDecoderRNN(attn,config['model']['embedding_dim'],tgt_vocab_size, config['model']['layer'])


    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Define the loss function + optimizer.
    loss_weights = torch.ones(tgt_vocab_size)
    loss_weights[0] = 0
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    criterion = torch.nn.NLLLoss(loss_weights)

    learning_rate = config['training']['learning_rate']
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                                        lr=learning_rate,
                                        weight_decay=config['training']['weight_decay'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(),
                                        lr=learning_rate,
                                        weight_decay=config['training']['weight_decay'])

    # Restore saved model (if one exists).
    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
        decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
    else:
        epoch = 0

    train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
    dev_log_string = '\n%s :: Epoch %i :: dev loss: %0.4f'
    clip = 5.0
    while epoch < config['training']['num_epochs']:

        # Main training loop.
        train_loss = []
        sampler = RandomSampler(train_data)
        for i, train_idx in enumerate(sampler):
            src, tgt = train_data[train_idx]

            # Clear gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Run words through encoder
            encoder_hidden = encoder.init_hidden()
            encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)

            # Feed desired outputs one by one from tgt into decoder
            # and measure loss.
            tgt_length = tgt.size()[0]
            loss = 0

            decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
            if USE_CUDA:
                decoder_context = decoder_context.cuda()

            for j in range(tgt_length - 1):
                decoder_input = tgt[j]
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                             decoder_context,
                                                                                             decoder_hidden,
                                                                                             encoder_outputs)
                loss += criterion(decoder_output, tgt[j+1])

            # Backpropagate the loss and update the model parameters.
            loss.backward()
            torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
            encoder_optimizer.step()
            decoder_optimizer.step()

            train_loss.append(loss.data.cpu())

            # Every once and a while check on the loss
            if ((i+1) % 100) == 0:
                print(train_log_string % (datetime.now(), epoch, i+1, len(train_data), np.mean(train_loss)), end='\r')
                train_loss = []

        # Evaluation loop.
        dev_loss = []
        for src, tgt in dev_data:
            # Run words through encoder
            encoder_hidden = encoder.init_hidden()
            encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)

            # Feed desired outputs one by one from tgt into decoder
            # and measure loss.
            tgt_length = tgt.size()[0]
            loss = 0

            decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
            if USE_CUDA:
                decoder_context = decoder_context.cuda()
            for j in range(tgt_length - 1):
                decoder_input = tgt[j]
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                             decoder_context,
                                                                                             decoder_hidden,
                                                                                             encoder_outputs)
                loss += criterion(decoder_output, tgt[j+1])
            dev_loss.append(loss.data.cpu())

        print(dev_log_string % (datetime.now(), epoch, np.mean(dev_loss)))

        state_dict = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict()
        }
        torch.save(state_dict, ckpt_path)

        epoch += 1

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True,
    #                     help='Configuration file.')
    # FLAGS, _ = parser.parse_known_args()
    # print(FLAGS)
    #FLAGS.config = 'config.yaml'
    main()

