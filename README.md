# [BasedLibrarian - Client](https://github.com/MaxSloof/BasedLibrarian-client/)

A little tool that lets you ask questions from your pdfs, epubs, text files and word documents. 

Available as a notebook that launches a simple web app with a UI. Uses `langchain` and `gradio` for much of the heavy lifting. 

# How to use

Put your documents in the 'docs/' folder

Create a 'configuration.yaml' file based on 'sample-configuration.yaml' 

Add your GCP credentials JSON to the './client' folder and reference the file location in the configuration file. 

Run the Docker Compose file using 
```bash
docker compose up
```

Go to 'http://localhost:7860'. 

Press 'Hard re-scan' when first deploying the container. 

BasedLibrarian will read your documents and generate embeddings for them. It will then use these embeddings to search for information relevant to your question, and pass this information to OpenAI to generate the response you see. If you add new documents while the program is running, hit the `Scan the library again` button. 

# Limitations and known bugs

Please note that because of how it does search, it's limited to direct and detailed questions; generic questions won't be of much use, and it won't act like a chatbot. 

Known issues:
- Container needs to be restarted after scanning newly added documents
- No input documents are provided to the LLM for inference. (Workaround: Press 'Hard re-scan')

# Future roadmap 
1. Add more Cloud providers
2. Add self-hosted options (see BasedLibrarian-server)
3. Add chat history to web-ui
4. Asynchronous function to 
