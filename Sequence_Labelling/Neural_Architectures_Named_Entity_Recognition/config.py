import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--char_embed', type=int, default=25)
parser.add_argument('--word_embed', type=int, default=100)
parser.add_argument('--char_hidden_size', type=int, default=25)
parser.add_argument('--word_hidden_size', type=int, default=100)
parser.add_argument('--train_file', type=str, default="./data/eng.train")
parser.add_argument('--val_file', type=str, default="./data/eng.testa")
parser.add_argument('--test_file', type=str, default="./data/eng.testb")
parser.add_argument('--pre_embed', type=str, default="./glove.6B.100d.txt")
parser.add_argument('--use_pre_embed', type=int, default=1)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--zeros', type=int, default=1)
parser.add_argument('--lower', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--crf', type=int, default=1)
parser.add_argument('--decay_factor', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--decay_size', type=float, default=1000)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--moving_average_decay', type=float, default=0.99)
parser.add_argument('--max_sent_len', type=int, default=100)

parser.add_argument('--model_dir', type=str, default='./saved')


args = parser.parse_args()
