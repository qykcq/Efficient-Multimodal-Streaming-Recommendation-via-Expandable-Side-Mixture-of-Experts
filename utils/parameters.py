import argparse
from engine import Engine


def construct_engine(configs):

    for config in configs:
        print(config)
        parser = add_params(config['dataset_name'])

        for key, val in config.items():
            parser.add_argument('--{}'.format(key), default=val, type=type(val))

        if config['dataset_name'] in ['hm']:
            parser.add_argument('--k_core', default=10, type=int)
        elif config['dataset_name'] in ['home', 'electronics']:
            parser.add_argument('--k_core', default=5, type=int)
        else:
            raise ValueError("Invalid choice of dataset!")
        args = parser.parse_args()
        engine = Engine(args)
        engine.train()


def add_params(dataset_name):
    parser = argparse.ArgumentParser(description='Argument parser for the program.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--num_experts', default=1, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--word_embedding_dim', default=768, type=int)
    parser.add_argument('--valid_portion', default=0.15, type=float)
    parser.add_argument('--time_windows', default=10, type=int)
    parser.add_argument("--num_words_title", type=int, default=40)
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--CV_resize', type=int, default=224)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--transformer_block', type=int, default=2)
    parser.add_argument('--adapter_activation', type=str, default="GELU")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument("--stored_vector_path", type=str, default=f"datasets/{dataset_name}/stored_vecs")
    parser.add_argument('--patience', type=str, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--adapter_down_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.95)
    return parser
