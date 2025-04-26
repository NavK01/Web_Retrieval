import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def create_vector_db(url):
    try:
        # Fetch website content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text()


        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content=text_content)] # Create document object
        texts = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)

        return db

    except requests.exceptions.RequestException as e:
        return [f"Error fetching website: {e}"]
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]



# get relevant text
def get_relevant_text(db, query, k=2):

    try:
        # Retrieve relevant text
        retriever = db.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.get_relevant_documents(query)

        return [doc.page_content for doc in retrieved_docs]
    
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]
    

def answer_query_with_context(query, context):

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto", torch_dtype=torch.float16)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
        )

        # Construct the prompt with context and query
        prompt = f"""Based on context, generate answer only.
                  Context: {context}\n\nQuery: {query}\n\nAnswer:"""

        response = pipe(prompt)
        return response[0]['generated_text'].replace(prompt, "").strip()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return "Error: CUDA out of memory. Try reducing batch sizes or using a smaller model."
        else:
            return f"An unexpected RuntimeError occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

#generate answer
# based on query and relevant text
# from vector db
def generate_answer(query, db):

    #search relevant text
    relevant_text_chunks = get_relevant_text(db, query)

    if relevant_text_chunks:
        answer = answer_query_with_context(query, relevant_text_chunks)
        return answer

    else:
        print("No relevant text chunks found.")

#create vector db
#website_url = "https://en.wikipedia.org/wiki/Cancer"
website_url = input("Enter the website URL: ")

db = create_vector_db(website_url)

while(1):
    user_query = input("Enter your question:  (1 to exit)")

    if (user_query == "1"):
        break

    answer = generate_answer(user_query, db)
    if answer == None:
        print("No answer found.")
        continue

    #print answer
    print("Question:::", user_query)
    print("*"*200)
    print("\n")
    print("Answer :==>", answer)
    print("\n")
    print("*"*200)