import logging

logging.basicConfig(format='%(levelname)s: %(name)s: %(asctime)s - [%(filename)s:%(lineno)d] %(message)s')

def get_logger(filename, level=logging.INFO):

    logger = logging.getLogger(filename)
    logger.setLevel(level)

    return logger