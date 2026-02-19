from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_pipeline import retrieve_documents


#this is creating the web server like what spring boot does with tomcat
app = FastAPI()


#this tells FastAPI that the incoming json must look like this, a variable named query with a value of string
# like meaning what you type as a request this is what it should look like  
class QueryRequest(BaseModel):
    query: str


#and then this simply takes that request and places itinto the function or calls the function 
#if someone sends POST request to this http, run the function below which is query_rag
@app.post("/rag/query")
#this says take the JSON sent in the request body and turn it into a python object
#meaning that this parameter is basically calling the Query request and is makign sure that the request the user 
# sent matches the contents of the function. When they do math, the variable inside on the left : uery, get a 
# request. put infront of it that holds the value of what the user asked, so tis basicalyl the variable query
# but since FASTapi is turning the json into code, this is how it turns the JSON into the variable code we are
#looking for. if instead of query i said yolo: str, this retrieve_documents would have request.yolo

#parameter checks if the user request matches the format of the variable we need,
#seperates it so that queury has the value of what the quesition the user asked it then we sned it 
def query_rag(request: QueryRequest):
    documents = retrieve_documents(request.query)

    return {
        "query": request.query,
        "documents": [doc.page_content for doc in documents] 
        }
