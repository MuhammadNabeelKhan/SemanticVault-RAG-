import os 

#so this module we import TextLoader and DirectoryLoader, these files will help us read files
from langchain_community.document_loaders import TextLoader, DirectoryLoader

#this will help us divide the large amounts of texts into chunks
#chunking is important because it saves space and llms are unable to process such huge documents and tokens
#all at once
from langchain_text_splitters import CharacterTextSplitter

#This will take the chunks of tokens (english words) and embeds them into a languge that the computer can read
from langchain_openai import OpenAIEmbeddings

#the chroma data based is held locally and it will store the embedded chunks
from langchain_chroma import Chroma

#this is just a helper module, it will help us take the api keys we will make in a envi file and retrieve them
from dotenv import load_dotenv

load_dotenv()

#this will load every txt file in our docs directory
def load_documents(docs_path= "docs"):

    print(f"Loading documents from {docs_path}...")
    
    #check if the directory even exists   
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory path: {docs_path} does not exist. Please enter a valid directory path to your documents.") 

    #so pretty self explanatory, path = the documents directory
    #glob is saying pick up all files that end with .txt
    #and we are using the langchain loader

    #now the important thing here is that directory loader basically has us avoid hardcoding loops that 
    #go through each file. We  give it a path, tell it what files to chase, and then
    #it will use textloader to open and read the file, and lastly makes a document object
    #consisting of the entire text and the metadata (the files path etc)
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader
        )
    
    #this will give us the list of the langchain document objects
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"The directory {docs_path} has no documents.")
    

    #every time i used doc in the loop it knows im referring to the ith index of documents
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents


def chunking(documents, chunk_size = 400, chunk_overlap = 0):
    print("Splitting documents into smaller chunks...")


    #This is basically a class that works with strings
    #it initalizes a object from a premade class in langchain that takes chunk size and chunk overlap
    #chunk size is how much tokens i.e letters i want in every chunk and 
    #chunk overlap is how much i want these chunks overlapping eachother for example
    #my name is
    #is nabeel
    #so 2 characters im overlapping
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
        )
    
    
    #here text_splitter is calling a function within the characterTextSplitter class
    chunks = text_splitter.split_documents(documents)

    #the reason an entirely seperate class is able to take document objects is because that class is desgined
    #to ask the attributes of the document object. In runtime, the document object is checked to have 
    #the text and the meta data which when we orignally created it in the code in the above function, it 
    #added the information to those attributes of the document object. Now since the document object 
    #was created that means those attributes were filled in and thats what we are calling
    #and the function above already worked, this means that we have access to the attribues of the 
    #document object

    return chunks


def embedding_storing_vectors(chunks, persist_directory = "db/chroma_db"):
    print("Embedding chunks and storing in ChromaDB vector store...")

    embedding_model = OpenAIEmbeddings(model= "text-embedding-3-small")

    #creating the vector storage 
    vector_store= Chroma.from_documents(
        documents = chunks,
        embedding= embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space" : "cosine"}
    )

    print("Vector Store has been created")

    return vector_store

def main():
    print("Main Function")
    documents = load_documents(docs_path="TxtDocs")
    document_chunks = chunking(documents)
    vector_store = embedding_storing_vectors(document_chunks)

#only run main if this file is being directly run not just being imported
if __name__ == "__main__":
    main()