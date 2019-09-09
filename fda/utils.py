import logging
import numpy as np

# some constants
LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'


def setup_logger(name, log_file, log_format=LOG_FORMAT, level=logging.INFO):
    """Function setup as many loggers as you want"""

    logging.basicConfig(filemode='w')
    formatter = logging.Formatter(log_format)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_gt_batch(gt_batch, graph_config):
    out = np.zeros((graph_config['batch_size'],
                    graph_config['num_classes']), dtype=np.float32)
    for ind, x in enumerate(gt_batch):
        out[ind, x + graph_config['offset']] = 1.0
    return out
