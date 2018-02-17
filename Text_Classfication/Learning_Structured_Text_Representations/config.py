import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--emb_str', type=int, default=25)
parser.add_argument('--emb_sem', type=int, default=75)
parser.add_argument('--sent_attention', type=str, default="sum")
parser.add_argument('--word_emb_size', type=int, default=200)
parser.add_argument('--max_sents', type=int, default=60)
parser.add_argument('--max_tokens', type=int, default=120)
parser.add_argument('--skip_gram', type=int, default=1)
parser.add_argument('--n_class', type=int, default=5)
parser.add_argument('--train_file', type=str, default="./data/yelp-2013.train")
parser.add_argument('--val_file', type=str, default="./data/yelp-2013.dev")
parser.add_argument('--test_file', type=str, default="./data/yelp-2013.test")
parser.add_argument('--save_path', type=str, default='./data/preprocessed-yelp.pkl')
parser.add_argument('--pre_embed', type=str, default="./glove.6B.100d.txt")
parser.add_argument('--rnn_type', type=str, default="lstm")
parser.add_argument('--use_pre_embed', type=int, default=1)
parser.add_argument('--drop_rate', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--decay_factor', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--decay_size', type=float, default=1000)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--moving_average_decay', type=float, default=0.99)

parser.add_argument('--model_dir', type=str, default='./saved')


args = parser.parse_args()