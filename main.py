import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from parameters import construct_engine

if __name__ == '__main__':

    configs = []

    config = {
        'dataset_name': 'hm',
        'layers': '6,12', 'tau': 0.1
    }
    configs.append(config)

    construct_engine(configs)


