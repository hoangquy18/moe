""" logger config file"""

import logging

def get_logger():
    """
        Logger 
    """
    
    logger = logging.getLogger("my-logger")
    
    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        file_handler = logging.FileHandler("my-logger.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger

logger = get_logger()