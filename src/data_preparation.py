import numpy as np
import pandas as pd
import src

from src.loggers import logger

def processing_files():
    logger.info("Loading the CSV files from data")
    df = pd.read_csv(src.FILE_NAME)

    logger.info("Saving files from dataframe")
    df.to_csv(src.NEW_FILE_NAME, encoding="utf-8", index= False)

    return True
