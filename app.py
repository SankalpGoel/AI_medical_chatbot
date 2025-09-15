from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os
import base64
import io
from PIL import Image

app = Flask(__name__)

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

# <--- MODIFICATION 1: Retrieve fewer documents ---
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

llm = ChatOpenAI(
    model="openai/gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-a0793a20e8679eaafe5c119f8a30d2997ca17e6280bcc762ffb7e303e01909fed",
    temperature=0.4,
    max_tokens=500)

rephrasing_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, rephrasing_prompt
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

def get_context(input_dict):
    return history_aware_retriever.invoke(input_dict)

rag_chain = (
    RunnablePassthrough.assign(
        context=get_context
    ).assign(
        answer=question_answer_chain
    )
)


@app.route("/")
def index():
    session.pop('chat_history', None)
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    # 1. Get Text and Image
    msg = request.form.get("msg", "")
    image_file = request.files.get("image")
    
    print(f"User Input: {msg}")

    # 2. Get and Convert History (with truncation)
    chat_history_serializable = session.get('chat_history', [])
    recent_history_serializable = chat_history_serializable[-2:] # Keep 1 turn
    
    chat_history_objects = []
    for item in recent_history_serializable:
        if item.get('role') == 'human':
            chat_history_objects.append(HumanMessage(content=item['content']))
        elif item.get('role') == 'ai':
            chat_history_objects.append(AIMessage(content=item['content']))

    # 3. Create the new HumanMessage
    human_message_content = msg

    if image_file:
        print("Image file detected. AGGRESSIVELY resizing...")
        try:
            image = Image.open(image_file.stream)
            
            # <--- MODIFIED: Resize to a tiny 256x256 ---
            max_size = (256, 256) 
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            image_format = image.format if image.format in ['JPEG', 'PNG'] else 'JPEG'
            
            if image_format == 'PNG':
                image.save(img_byte_arr, format='PNG')
            else:
                # <--- MODIFIED: Drop quality to 50 ---
                image.save(img_byte_arr, format='JPEG', quality=50) 

            img_bytes = img_byte_arr.getvalue()
            
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            mime_type = image_file.mimetype or 'image/jpeg'
            image_url_string = f"data:{mime_type};base64,{img_base64}"
            
            human_message_content = [
                {"type": "text", "text": msg},
                {"type": "image_url", "image_url": {"url": image_url_string}}
            ]
            print("Image resized and encoded.")

        except Exception as e:
            print(f"Error processing image: {e}")
            human_message_content = msg + "\n(System: There was an error processing the uploaded image.)"
    
    human_message = HumanMessage(content=human_message_content)

    # 4. Call the chain
    response = rag_chain.invoke({
        "input": msg,
        "question": human_message.content,
        "chat_history": chat_history_objects
    })
    answer = response["answer"]
    
    print(f"Response: {answer}")

    # 5. Save history
    chat_history_serializable.append(
        {"role": "human", "content": human_message.content}
    )
    chat_history_serializable.append(
        {"role": "ai", "content": answer}
    )
    
    session['chat_history'] = chat_history_serializable[-2:]

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)