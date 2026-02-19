from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"


#this ai model was heavily trained on semeantics and word connection through a plethora of documents, and it 
#maps text in vector dimension of  around 1500
embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space" : "cosine" }
)


# query = "when was wikipedia invented?"

retriever = db.as_retriever(search_kwargs = {"k": 5} )

"""
retriever = db.as_retriever(  
    search_type = "similarity_score_threshold"
    search_kwargs = {
        "k" : 5
        "score_threshold" : 0.3
    }
)
"""


#we are making a function here to our RAG ai becomes callable 


#what we have in the function is what we call a type hint, alone they do nothing 
#but by using fastapi or mypy we are able to see clearly/neatly what the issues could be with the function.
def retrieve_documents(query: str) -> list: 
    relevant_docs = retriever.invoke(query)
    return relevant_docs



#so this is just to test it for us, we open everything
# print(f"user query is {query}")

# for i, doc in enumerate(relevant_docs):
#     print(f" Document {i + 1}: \n {relevant_docs[i]} \n")