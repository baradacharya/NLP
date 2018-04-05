"""Model evaluation script.

Usage:
    python evaluate.py --config config.yaml
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
import torch
import yaml
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable

from model_attention import EncoderRNN, Attn, AttnDecoderRNN
from utils import ShakespeareDataset, Vocab


FLAGS = None


class GreedyTranslator(object):
    def __init__(self,
                 encoder,
                 decoder,
                 tgt_vocab,
                 max_length=80):
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = tgt_vocab.word2id(tgt_vocab.sos_token)
        self.eos_id = tgt_vocab.word2id(tgt_vocab.eos_token)
        self.max_length = max_length

    def __call__(self, src):
        # Run words through encoder
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(src, encoder_hidden)

        # Greedily translate.
        decoder_input = Variable(torch.LongTensor([self.sos_id]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        translation = [self.sos_id]
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        for di in range(self.max_length):

            # Feed data into decoder.
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)



            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data


            # Find most likely word id.
            _, word_id = decoder_output.data.topk(1)
            word_id = word_id[0][0]

            translation.append(word_id)
            if word_id == self.eos_id:
                break

            # Convert word id to tensor to be used as next input to the decoder.
            decoder_input = Variable(torch.LongTensor([word_id]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        return translation, decoder_attentions[:di + 1, :len(encoder_outputs)]


def main(matrix = False):
    # Load the configuration file.
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    # Load the vocabularies.
    src_vocab = Vocab.load(config['data']['src']['vocab'])
    tgt_vocab = Vocab.load(config['data']['tgt']['vocab'])

    # Load the training and dev datasets.
    test_data = ShakespeareDataset('test', config, src_vocab, tgt_vocab)

    # Restore the model.
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    import pickle
    if matrix:
        f = open('attention_mat.pkl', 'rb')
        attention_matrix = pickle.load(f)
        f.close()
        for i in range(10):
            src, tgt = test_data[i]
            decoder_attn = attention_matrix[i]
            src_sentence = [src_vocab.id2word(id) for id in src.data.cpu().numpy()]
            tgt_sentence = [tgt_vocab.id2word(id) for id in tgt.data.cpu().numpy()]

            src_sentence_ = ' '.join(src_sentence)
            tgt_sentence_ = ' '.join(tgt_sentence)
            show_attention(src_sentence_, tgt_sentence_, decoder_attn)
        return

    encoder = EncoderRNN(src_vocab_size, config['model']['embedding_dim'],config['model']['layer'])
    attn = 'general'
    decoder = AttnDecoderRNN(attn,config['model']['embedding_dim'],tgt_vocab_size, config['model']['layer'])

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    ckpt_path = os.path.join(config['data']['ckpt'], config['experiment_name'], 'model.pt')
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
    else:
        print('Unable to find checkpoint. Terminating.')
        sys.exit(1)
    encoder.eval()
    decoder.eval()

    # Initialize translator.
    greedy_translator = GreedyTranslator(encoder, decoder, tgt_vocab)

    # Qualitative evaluation - print translations for first couple sentences in
    # test corpus.

    import numpy as np
    attention_matrix = []
    import pickle
    for i in range(10):
        src, tgt = test_data[i]
        translation,decoder_attn = greedy_translator(src)
        attention_matrix.append(decoder_attn.numpy())
        src_sentence = [src_vocab.id2word(id) for id in src.data.cpu().numpy()]
        tgt_sentence = [tgt_vocab.id2word(id) for id in tgt.data.cpu().numpy()]
        translated_sentence = [tgt_vocab.id2word(id) for id in translation]
        print('---')
        print('Source: %s' % ' '.join(src_sentence))
        print('Ground truth: %s' % ' '.join(tgt_sentence))
        print('Model output: %s' % ' '.join(translated_sentence))
    print('---')
    f = open('attention_mat.pkl', 'wb')
    pickle.dump(attention_matrix, f)
    f.close()

    # Quantitative evaluation - compute corpus level BLEU scores.
    hypotheses = []
    references = []
    for src, tgt in test_data:
        translation,decoder_attn = greedy_translator(src)
        tgt_sentence = [tgt_vocab.id2word(id) for id in tgt.data.cpu().numpy()]
        translated_sentence = [tgt_vocab.id2word(id) for id in translation]
        # Remove start and end of sentence tokens.
        tgt_sentence = tgt_sentence[1:-1]
        translated_sentence = translated_sentence[1:-1]
        hypotheses.append(tgt_sentence)
        references.append([translated_sentence])
    print("Corpus BLEU score: %0.4f" % corpus_bleu(references, hypotheses))

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words.split(' '))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True,
    #                     help='Configuration file.')
    # FLAGS, _ = parser.parse_known_args()

    main(matrix = True)

