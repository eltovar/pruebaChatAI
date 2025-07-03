import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import openai
from datetime import datetime

# LangChain imports para RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain imports para Agente y Tools
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory # IMPORTANTE: Para mantener el contexto de la conversación
from langchain.prompts import MessagesPlaceholder # Para usar en el prompt del agente con memoria
  pip install langchain==0.3.26 langchain-community==0.3.27 flask-socketio==5.5.1
# Flask-SocketIO para notificaciones en tiempo real
from flask_socketio import SocketIO, emit

from langchain.prompts import PromptTemplate

# --- Cargar variables de entorno al iniciar la aplicación ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada.")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configuración e inicialización del sistema RAG ---
VECTOR_STORE_PATH = "faiss_index_mi_conocimiento"
DOCUMENT_PATH = "data/documento.txt"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(VECTOR_STORE_PATH):
    print(f"Cargando índice FAISS desde: {VECTOR_STORE_PATH}")
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print(f"Creando nuevo índice FAISS y guardándolo en: {VECTOR_STORE_PATH}")
    os.makedirs(os.path.dirname(DOCUMENT_PATH), exist_ok=True)
    try:
        with open(DOCUMENT_PATH, encoding="utf-8") as f:
            texto = f.read()
    except FileNotFoundError:
        print(f"Error: El archivo de documento '{DOCUMENT_PATH}' no se encontró. Crea un archivo.")
        texto = "No se encontraron documentos de conocimiento para RAG."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.create_documents([texto])

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(VECTOR_STORE_PATH)

rag_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=rag_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

def responder_con_rag(pregunta: str) -> str:
    """Función para responder a preguntas usando el sistema RAG."""
    try:
        result = qa_chain.invoke({"query": pregunta})
        return result.get("result", "No se encontró información relevante en los documentos.")
    except openai.APIError as e:
        print(f"Error de API de OpenAI en RAG: {e}")
        return "Lo siento, hubo un problema con el servicio de IA al buscar información."
    except Exception as e:
        print(f"Error al ejecutar RAG: {e}")
        return f"Ocurrió un error inesperado al procesar tu solicitud: {e}"

agent_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

# --- Definición de Herramientas (Tools) para el Agente ---
@tool
def get_current_weather(location: str) -> str:
    """Obtiene el clima actual para una ubicación dada."""
    if "Medellín" in location or "Itagüí" in location or "Envigado" in location:
        return "El clima actual en el Valle de Aburrá (Medellín/Itagüí) es soleado con una temperatura de 30°C. Ideal para actividades al aire libre."
    elif "Bogotá" in location:
        return "El clima en Bogotá es nublado con 18°C. Hay una probabilidad de lluvias ligeras en la tarde."
    elif "Cali" in location:
        return "El clima en Cali es parcialmente nublado con 27°C. Se siente cálido y húmedo."
    else:
        return f"No tengo información del clima para {location} en este momento. Por favor, especifica una ciudad principal de Colombia."

@tool
def process_knowledge_query(query: str) -> str:
    """Procesa una consulta que requiere información de la base de conocimiento (RAG).
    Útil cuando el usuario pregunta sobre detalles específicos de productos, políticas, o temas internos."""
    rag_response = responder_con_rag(query)
    if "No se encontró información relevante" in rag_response or "Ocurrió un error inesperado" in rag_response:
        return "NO_INFO_ENCONTRADA_RAG"
    return rag_response

@tool
def finalize_sale_process(product_info: str, client_info: str) -> str:
    """Simula la finalización de un proceso de venta y emite una notificación.
    Útil cuando el agente ha confirmado todos los detalles necesarios para registrar una venta."""
    print(f"DEBUG: Venta finalizada para {client_info} de {product_info}. Enviando notificación...")

    sale_data = {
        "status": "OK",
        "message": f"¡Nueva venta generada: {product_info} para {client_info}!",
        "product": product_info,
        "client": client_info,
        "timestamp": datetime.now().isoformat()
    }
    socketio.emit('new_sale_notification', sale_data)
    print("DEBUG: Notificación de venta emitida vía WebSocket.")

    return f"¡Excelente! El proceso de venta de {product_info} para {client_info} ha sido exitoso y se ha generado la notificación."

tools = [
    get_current_weather,
    process_knowledge_query,
    finalize_sale_process
]

# --- Inicializar la memoria de conversación ---
# IMPORTANTE: En este setup de Flask sin estado persistente para cada usuario,
# esta memoria se reiniciará con cada nueva solicitud de webhook.
# Para persistencia real entre llamadas, necesitas guardar/cargar la memoria en una DB (ej., Redis, Firestore).
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Inicializar el Agente de LangChain ---
# Cargamos el prompt base de ReAct.
prompt = hub.pull("hwchase17/react")

# Modificamos el prompt para incluir el historial de chat.
# Usamos `MessagesPlaceholder` para insertar dinámicamente el historial.
# Asegúrate de que tu `prompt` incluye esta sección.
# Un prompt base de ReAct no siempre incluye un chat_history. Tendrías que crearlo o modificarlo.
# Ejemplo de cómo se vería un prompt extendido para manejar memoria:
# Ver el ejemplo de "agent with memory" en la documentación de LangChain si quieres un prompt más completo.

# Una forma de añadir chat_history al prompt si el hub.pull no lo tiene por defecto:
# Si el prompt base no incluye ya un placeholder para chat_history, necesitamos crearlo.
# Un buen ejemplo sería usar `AgentExecutor.from_agent_and_tools` con un prompt personalizado
# que incluya `chat_history`.

# Para un agente ReAct simple con memoria, el `prompt` debería tener una variable `chat_history`.
# Si `hub.pull("hwchase17/react")` no la tiene, podemos definirla:
prompt_with_history = PromptTemplate.from_template(
    """Actúa como un asistente útil.
Tu historial de conversación:
{chat_history}
Pregunta actual: {input}
{agent_scratchpad}"""
)


agent = create_react_agent(agent_llm, tools, prompt_with_history) # Usamos el prompt con historial
# Añadimos la memoria al AgentExecutor.
# `memory_key="chat_history"` debe coincidir con el nombre de la variable en el prompt.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory, # ¡Aquí integramos la memoria!
    handle_parsing_errors=True
)

# --- Webhook de Flask para Dialogflow ---
@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    print(f"Dialogflow Request Body: {json.dumps(req, indent=2)}")

    query_result = req.get('queryResult', {})
    query_text = query_result.get('queryText', '').strip()
    intent_display_name = query_result.get('intent', {}).get('displayName')
    parameters = query_result.get('parameters', {})
    session_id = req.get('session') # Obtener el ID de sesión de Dialogflow

    fulfillment_text = "Lo siento, no pude procesar tu solicitud."

    try:
        # --- Lógica para manejar el menú estático de Dialogflow ---
        if intent_display_name == 'MenuOpcion_Horarios':
            fulfillment_text = "Nuestro horario de atención es de lunes a viernes de 8:00 AM a 5:00 PM y sábados de 9:00 AM a 1:00 PM."
        elif intent_display_name == 'MenuOpcion_Contacto':
            fulfillment_text = "Puedes contactarnos al 018000-123456 o enviarnos un correo a info@ejemplo.com."
        elif intent_display_name == 'MenuOpcion_Soporte':
            fulfillment_text = "Para soporte técnico, por favor detalla tu problema. Un agente de IA o humano te asistirá. O puedes visitar nuestra sección de preguntas frecuentes en el sitio web."
        # --- Lógica para la venta (si Dialogflow ha confirmado todos los parámetros) ---
        elif intent_display_name == 'Confirmar_Venta_Intent':
            product = parameters.get('producto')
            client = parameters.get('cliente')
            if product and client:
                fulfillment_text = finalize_sale_process(product, client)
            else:
                fulfillment_text = "Necesito más información (producto y cliente) para finalizar la venta."
        # --- Si no es una opción de menú o venta confirmada, pasar al Agente de LangChain ---
        else:
            # Aquí la memoria se reiniciará con cada solicitud del webhook a menos que
            # la persistas externamente usando el session_id.
            # Para una demostración local, funciona para mantener el contexto *dentro* de la única invocación del agente.
            response_agent = agent_executor.invoke({"input": query_text})
            agent_output = response_agent.get("output", "No pude generar una respuesta con el agente.")

            if "NO_INFO_ENCONTRADA_RAG" in agent_output:
                fulfillment_text = f"Lo siento, no tengo información específica sobre '{query_text}' en mi base de conocimientos. ¿Hay algo más en lo que pueda ayudarte o te gustaría hablar con un asesor?"
            else:
                fulfillment_text = agent_output

    except openai.APIError as e:
        print(f"Error de API de OpenAI: {e}")
        fulfillment_text = "Lo siento, hubo un problema con el servicio de IA. Por favor, inténtalo más tarde."
    except Exception as e:
        print(f"Error general en el procesamiento del webhook: {e}")
        fulfillment_text = f"Lo siento, algo salió mal con el asistente. Error: {e}. Por favor, inténtalo de nuevo."

    return jsonify({
        'fulfillmentText': fulfillment_text
    })

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def test_connect():
    print('Cliente WebSocket conectado!')

@socketio.on('disconnect')
def test_disconnect():
    print('Cliente WebSocket desconectado.')

if __name__ == '__main__':
    print("Iniciando Flask con SocketIO...")
    socketio.run(app, port=8080, debug=True, allow_unsafe_werkzeug=True)