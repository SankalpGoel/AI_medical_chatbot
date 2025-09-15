from flask import Flask, render_template, jsonify, request, session  # <--- NEW: Import session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever  # <--- NEW: Import history-aware retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <--- NEW: Import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage # <--- NEW: Import message types
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# <--- NEW: You MUST set a secret key to use Flask sessions
# It's best to set this as an environment variable
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_secure_default_key_12345')

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "ai-medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="openai/gpt-5-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-a0793a20e8679eaafe5c119f8a30d2997ca17e6280bcc762ffb7e303e01909fed",
    temperature=0.4,
    max_tokens=500)

#Prompt for rephrasing the question based on history
rephrasing_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation")
])

#This retriever takes history and the new question, rephrases it, and *then* retrieves docs
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, rephrasing_prompt
)

# The final prompt now also accepts chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


@app.route("/")
def index():
    #Clear the chat history when the user loads the main page
    session.pop('chat_history', None)
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Input: {input}")

    # 1. Get the JSON-safe list of dictionaries from the session
    chat_history_serializable = session.get('chat_history', [])
    
    # 2. Convert the list of dicts back into HumanMessage/AIMessage objects
    #    This is what the RAG chain expects
    chat_history_objects = []
    for item in chat_history_serializable:
        if item.get('role') == 'human':
            chat_history_objects.append(HumanMessage(content=item['content']))
        elif item.get('role') == 'ai':
            chat_history_objects.append(AIMessage(content=item['content']))

    # 3. Call the chain with the *objects*
    response = rag_chain.invoke({
        "input": msg, 
        "chat_history": chat_history_objects
    })
    answer = response["answer"]
    
    print(f"Response: {answer}")

    # 4. Update the JSON-safe list (not the objects)
    chat_history_serializable.append({"role": "human", "content": msg})
    chat_history_serializable.append({"role": "ai", "content": answer})
    
    # 5. Save the JSON-safe list back to the session
    session['chat_history'] = chat_history_serializable

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)