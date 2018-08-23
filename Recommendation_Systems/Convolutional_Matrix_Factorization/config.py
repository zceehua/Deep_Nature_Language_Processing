import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lambda_u', type=int, default=1)
parser.add_argument('--lambda_v', type=int, default=100)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--embedding', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--model_dir', type=str, default='./saved')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_filters', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10)



args = parser.parse_args()