import gradio as gr
import time

from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings


from langchain.llms import VertexAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

import vector_db
import prompt_template
import file_loader

# Document directory
DOC_DIR = "./books/"

# Vector db
db = Chroma(embedding_function=VertexAIEmbeddings(), client_settings=vector_db.CHROMA_SETTINGS)

# Google Cloud API
llm = VertexAI(temperature=0,max_output_tokens=256)

# Langchain
chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=prompt_template.QUESTION_PROMPT)

# Embed the docs
def embed_docs(progress=gr.Progress()):
    """
    Scans new documents added to the folder
    """
    global db

    progress(0.1, desc="Scanning for new documents ...")
    # Cuts all docs into chunks of 1000 tokens
    texts = file_loader.process_text(DOC_DIR)

    if texts != None:
        progress(0.5,desc="Embedding documents ...")
        embeddings = VertexAIEmbeddings()
        db.add_documents(texts, embedding=embeddings, persist_directory=vector_db.PERSIST_DIRECTORY)
        # Save all docs in the folder 
        docs_in_folder = file_loader.docs_in_folder(DOC_DIR)
    if texts == None:
        progress(0.9, desc="No new documents found!")
        time.sleep(1)
   
def hard_embed_docs(progress=gr.Progress()):
    """
    Embeds all the documents in the vector db from scratch
    """
    global db

    progress(0.3, desc="Clearing collection ...")

    db.delete_collection()
    vector_db.delete_database()

    db = Chroma(embedding_function=VertexAIEmbeddings(), client_settings=vector_db.CHROMA_SETTINGS)

    progress(0.4, desc="Embedding documents ...")
    embed_docs()

# Main query code.
def ask(query, selected_docs, progress=gr.Progress()):
    start_time = time.perf_counter()

    # Transform selected documents into Chroma filter format
    if selected_docs == docs_in_folder:
        search_filter = None
    elif selected_docs != docs_in_folder and len(selected_docs) > 1:
        source_list = [{'source': doc} for doc in selected_docs]
        search_filter = {"$or": source_list}
    elif selected_docs != docs_in_folder and len(selected_docs) == 1:
        search_filter = {'source': selected_docs[0]}

    progress(0.1, desc="Scanning embedded documents for matches ...")
  
    #generate docs which are texts relevant to the query
    docs = db.similarity_search(query, filter=search_filter) 

    docs = vector_db.check_metadata_page(docs)

    progress(0.2, desc="Assembling request...")
    progress(0.4, desc="Appending citations and metadata ...")
    #some muckwork to log sources
    x = 0
    citations = ""
    for x in range(len(docs)): 
        # citations += docs[x].metadata['source']  + "\n"
        citations += docs[x].metadata['source'] + " in page: " + str(docs[x].metadata['page'])  + "\n"
    
    progress(0.5, desc="Talking to LLM for answers ...")

    #calls llm with the query and relevant docs in hand and returns both the response and the sources    
    librarianoutput = chain({"input_documents":docs, "question": query})
    output = "Answer: \n" + librarianoutput["output_text"] + "\n\nI found this in: \n" + citations

    query_time = str(round(time.perf_counter() - start_time ,2)) + " seconds"
    print(query_time)
    return output

# Save all docs in the folder 
docs_in_folder = file_loader.docs_in_folder(DOC_DIR)

#Gradio UI
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("# Welcome to your BasedLibrarian!")
        scan_btn = gr.Button("Scan the library again.")
        hard_embed_btn = gr.Button("Hard re-scan the library again.")
        
    query = gr.Textbox(label="What can I help you find?")
    output = gr.Textbox(label="Response:")
    ask_btn = gr.Button("Ask Librarian")
    # performance_box = gr.Textbox(label=f"Time to complete query:",)
    checkbox_docs = gr.CheckboxGroup(docs_in_folder,value = file_loader.docs_in_folder(DOC_DIR), label="Documents used as input for the model",every=5)

    ask_btn.click(fn=ask, inputs=[query, checkbox_docs], outputs=output)
    scan_btn.click(fn=embed_docs, outputs=output)
    hard_embed_btn.click(fn=hard_embed_docs, outputs=output)

app.queue(concurrency_count=1).launch(server_name="0.0.0.0")