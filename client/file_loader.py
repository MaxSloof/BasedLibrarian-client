import glob

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

embed_docs_dir = "/embed_docs.txt"

def process_text(DOC_DIR):
    library = create_library(DOC_DIR)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = []
    for file_name in library:
          
          # Splits all documents into chunks of 1000 characters
          text = text_splitter.split_documents(library[file_name])
          texts += text

          # Save the file name of the document to a txt file that will be used to check whether the documents has already been added to the vector database
          safe_embed_docs(file_name)
    
    # If no docs have been added, return None
    if texts == []: return None

    return texts

# Supported file extensions
extensions = ['txt', 'md', 'pdf', 'doc', 'docx']
def create_library(DOC_DIR):
            
    # Initialize library as dict and load documents into it
    library = {}
    for extension in extensions:
        for file_name in glob.glob(f"{DOC_DIR}*." + extension):
            # Checks if doc has already been embedded
            if check_embed_docs(file_name):
              continue
            if extension in ['txt', 'md']:
                loader = TextLoader(file_name)
            elif extension in ['doc', 'docx']:
                loader = UnstructuredWordDocumentLoader(file_name) 
            elif extension == 'pdf':
                loader = PyPDFLoader(file_name)
            elif extension == 'epub':  
                loader = UnstructuredEPubLoader(file_name) 
            
            
            # Load the contents of the file        
            documents = loader.load()
            # Add the file and its contents to the library
            library[file_name] = documents

    return library

def safe_embed_docs(file_name):
    """
    Saves the file name of the document to a txt file that will be used to check whether the documents has already been added to the vector database
    """
    try:
        with open('../db/embed_docs.txt', 'a') as f:
            f.write(file_name + '\n')
    except FileNotFoundError:
        with open('../db/embed_docs.txt', 'w') as f:
            f.write(file_name + '\n')

def check_embed_docs(file_name):
    """
    Checks whether the document has already been added to the vector database
    """
    try:
        with open(f'../db/embed_docs.txt', 'r') as f:
            for line in f:
                if line.strip() == file_name:
                    return True
    except FileNotFoundError:
        return False
    return False

def clear_embed_docs_file():
    """
    Clears the embed_docs.txt file
    """
    with open('../db/embed_docs.txt', 'w') as f:
        f.write('')

def docs_in_folder(DOC_DIR):
    """
    Returns the number of documents in the folder
    """
    docs_in_folder = []
    for extension in extensions:
        for file_name in glob.glob(f"{DOC_DIR}*." + extension):
            docs_in_folder.append(file_name)
    
    return docs_in_folder
