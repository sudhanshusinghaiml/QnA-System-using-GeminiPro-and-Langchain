FILE_NAME = "faqs_data.csv"
NEW_FILE_NAME = "faqs_data_new.csv"
MODEL_NAME = "hkunlp/instructor-large"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': True}
VECTORDB_FILE_PATH = "vector_index.pkl"