from chromadb.config import Settings
import os

import file_loader
# Persistent Vector DB directory
PERSIST_DIRECTORY = "../db"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


def check_metadata_page(docs):
    for i in range(len(docs)):
        if 'page' not in docs[i].metadata.keys():
            docs[i].metadata['page'] = "not applicable"
    return docs


def delete_database():
    # Delete all files in the persist directory
    for file in os.listdir(PERSIST_DIRECTORY):
        file_path = os.path.join(PERSIST_DIRECTORY, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            if os.path.isdir(file_path):
                for file_2 in os.listdir(file_path):
                    file2_path = os.path.join(file_path, file_2)
                    try:
                        if os.path.isfile(file2_path):
                            os.remove(file2_path)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)

    file_loader.clear_embed_docs_file()
