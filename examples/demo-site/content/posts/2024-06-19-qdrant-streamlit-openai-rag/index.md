+++
title = "Building my own chatgpt agent replica on OpenAI's GPT-4"
date = 2024-06-19
draft = false
[taxonomies]
    tags = ["openai", "gpt-4", "rag", "qdrant", "streamlit"]
+++

## Why?

[OpenAI](https://openai.com/) has been making it easier and easier to build out [GPT agents](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/) that make use of your own data to improve the generated responses of the pretrained models.

Agents give a way to inject knowledge about your specific proprietary data into your pipeline, without actually sharing any private information about it. You can also improve the recency of your data too which makes you less dependent on the model's training cycle.

OpenAI has improved the DX, UX and APIs since version 3.5, and has made it easier to create `agents` and embed your data into your custom [`GPTs`](https://openai.com/index/introducing-gpts/). They have lowered the barrier to entry which means that virtually anyone can build their own assistants that would be able to respond to queries about their data. This is perfect for people to experiment on building products. IMO this is a very good approach to enable product discovery for the masses.

Most big AI contenders on the market provide you with a toolbox of high level abstractions and low to no code solutions. The weird thing about my approach to learning things is that not having some understanding of the first principles of the tech I'm using makes me feel a bit helpless, this is why I figured trying to build my own `RAG` system would be a good way to figure out the nuts and bolts.

## What?

I wanted to get a project for running my own pipeline with somewhat interchangeable parts. Models can be swapped around so that you can make the most of the latest models either available on [`Hugginface`](https://huggingface.co/), [`OpenAI`](https://openai.com/) or wherever.

Because things are moving so fast in model research the top contenders are surpassing each other every day pretty much. A custom pipeline  would allow us to quickly iterate and test out new models as they evolve. This allows you to try out new models and just as easily rollback your experiment.

What I wound up building is a [`Streamlit`](https://streamlit.io/) app that uses [`qdrant`](https://qdrant.com/) to index and search data extracted from a collection of `pdf` document. The app is a simple chat interface where you can ask questions about the data and get responses from a mixture of `GPT-4` and the indexed data.

## How?

#### 1. Setting up the environment
   - use `pyenv` to manage python versions
   ```bash
   # update versions
   pyenv update
   # install any python version
   pyenv install 3.12.3 # as of writing this
   # create a virtualenv
   ~/.pyenv/versions/3.12.3/bin/python -m venv .venv
   # and then activate it
   source .venv/bin/activate
   ```
#### 2. Install the dependencies
   ```bash
   # install poetry
   pip install poetry
   # install the dependencies
   poetry install
   ```
   the dependencies section of the `pyproject.toml` file should look like this:
   ```toml
   ...
   [tool.poetry.dependencies]
    python = "^3.12"
    streamlit = "^1.32.1"
    langchain = "^0.1.12"
    python-dotenv = "^1.0.1"
    qdrant-client = "^1.8.0"
    openai = "^1.13.3"
    huggingface-hub = "^0.21.4"
    pydantic-settings = "^2.2.1"
    pydantic = "^2.6.4"
    pypdf2 = "^3.0.1"
    langchain-community = "^0.0.28"
    langchain-core = "^0.1.31"
    langchain-openai = "^0.0.8"
    instructorembedding = "^1.0.1"
    sentence-transformers = "2.2.2"
   ...
   ```

#### 3. Set up the loading of the variables from a config file
   - a nice way to manage settings is to use `pydantic` and `pydantic-settings`
   ```python
   from pydantic import Field, SecretStr
   from pydantic_settings import BaseSettings, SettingsConfigDict

   class Settings(BaseSettings):
       model_config = SettingsConfigDict(env_file="config.env", env_file_encoding="utf-8")
       hf_access_token: SecretStr = Field(alias="HUGGINGFACEHUB_API_TOKEN")
       openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")

   ```
   this way you can load the settings from `config.env` but variables in the environment override the ones in the file.

   - a nice extra is that you also get type checking and validation from `pydantic` including `SecretStr` types for sensitive data.


#### 4. Set up the UI elements

   - Streamlit makes it quite easy to strap together a layout for your app. You have a single script that can run via the streamlit binary:
   ```bash
   streamlit run app.py
   ```
   [The gallery](https://streamlit.io/components?category=all) has many examples of various integrations and components that you can use to build your app. You have smaller components like inputs and buttons but also more complex UI tables, charts, you even have [`ChatGPT`](https://streamlit.io/components?category=llms) style templates.

   For our chat interface we require very few elements. Generally to create them you only need to use streamlit to initialize the UI.
   ```python
   import streamlit as st
   ...
   def main():
       st.title("ChatGPT-4 Replica")
       st.write("Ask me anything about the data")
       question = st.text_input("Ask me anything")
       if st.button("Ask"):
           st.write("I'm thinking...")
           response = get_response(question)
           st.write(response)
   ...
   main()
   ```

   The one thing I find a bit awkward is the fact that if you have elements that need to be conditionally displayed the conditions tend to resemble the javascript pyramid of doom if you have too many conditionals in the same block.

   Below is a simple example so you can see what I mean:
   ```python
   if len(pdf_docs) == 0:
       st.info("Please upload some PDFs to start chatting.")
   else:
       with st.sidebar:
           if st.button("Process"):
               with st.spinner("Processing..."):
                   # get raw content from pdf
                   raw_text = get_text_from_pdf(pdf_docs)
                   text_chunks = get_text_chunks(raw_text)

                   if "vector_store" not in st.session_state:
                       start = time.time()
                       st.session_state.vector_store = get_vector_store(text_chunks)
                       end = time.time()
                       # create vector store for each chunk
                       st.write(f"Time taken to create vector store: {end - start}")
   ```

   This makes me think that it is probably not designed for complex UIs but rather for quick prototyping and simple interfaces.
   

#### 5. pdf data extraction
   
   - I used the `PyPDF2` library to extract the text from the pdfs. The library is quite simple to use and you can extract the text from a pdf file with a few lines of code.
   ```python
   import PyPDF2

   def get_text_from_pdf(pdf_docs):
       raw_text = ""
       for pdf in pdf_docs:
           pdf_file = pdf["file"]
           pdf_reader = PyPDF2.PdfFileReader(pdf_file)
           for page_num in range(pdf_reader.numPages):
               page = pdf_reader.getPage(page_num)
               raw_text += page.extract_text()
       return raw_text
   ```

   - The extracted text should be chunked into smaller pieces that can be used to create embeddings for the `qdrant` index.
   ```python
   def get_text_chunks(raw_text):
       text_chunks = []
       for i in range(0, len(raw_text), 1000):
           text_chunks.append(raw_text[i:i + 1000])
       return text_chunks
   ```

#### 6. Setting up the `qdrant` server via `docker`
   The best way to set up `qdrant` is to use docker and to keep track of the environment setup `docker-compose` is a nice approach. You can set up the `qdrant` server with a simple `docker-compose.yml` file like the one below:
   ```yaml
   version: '3.9'

   services:
     qdrant:
       image: qdrant/qdrant:latest
       ports:
         - "6333:6333" # Expose Qdrant on port 6333 of the host
       volumes:
         - qdrant_data:/qdrant/data # Persistent storage for Qdrant data
       environment:
         RUST_LOG: "info" # Set logging level to info

   volumes:
     qdrant_data:
       name: qdrant_data
   ```


#### 7. Indexing the data
   - The `qdrant` client can be used to index the embeddings and perform similarity search on the data. You can pick and choose the best model for embeddings for your data and swap them out if you find [a better one](https://huggingface.co/spaces/mteb/leaderboard).
   ```python
   def get_vector_store(text_chunks, qdrant_url="http://localhost:6333"):
       embeddings = HuggingFaceInstructEmbeddings(model_name="avsolatorio/GIST-Embedding-v0", model_kwargs={"device": "mps"})
       vector_store = Qdrant.from_documents(
           text_chunks,
           embeddings,
           url=qdrant_url,
           collection_name="pdfs",
           force_recreate=True,
       )
       return vector_store
   ```

#### 8. sending the query 
   In order to send the query to `qdrant` you again need to embed it to allow to do a similarity search over your collection of documents.
   ```python
   def get_response(question, qdrant_url="http://localhost:6333"):
        embeddings = HuggingFaceInstructEmbeddings(model_name="avsolatorio/GIST-Embedding-v0", model_kwargs={"device": "mps"})
        query_vector = embeddings.encode(question)
        vector_store = Qdrant(url=qdrant_url, collection_name="pdfs")
        response = vector_store.search(query_vector, top_k=1)
        return response
   ```

#### 9. Analysis
   You can swap out any of the components in this project with something else. You could use [`Faiss`](https://github.com/facebookresearch/faiss) instead of `qdrant`, you could use `OpenAI` models for everything(embeddings/chat completion) or you could use open models.

   You can forego the UI and simply use `fastapi` to create an API to interact with the PDF documents. I hope this gives you some sense of the possibilities that are available to you when building your own `RAG` system.

## Conclusions
- you can build your own agent and have it respond to queries about your data quite easily
- `streamlit` is great for prototyping and building out simple interfaces
- `qdrant` is good for performing similarity search on your data
- when building `RAG` systems you need to make use of embedding models to encode your data
- embedding models are the most taxing parts of the pipeline
- if you have pluggable parts in your pipeline you can swap them out easily to save costs
- `pydantic` and `pydantic-settings` are great for adding type checking and validation to your python code

