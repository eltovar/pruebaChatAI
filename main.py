import os
import json
from flask import Flask, request, jsonify # send_from_directory no es necesario si solo es API
from dotenv import load_dotenv
import openai
from datetime import datetime
import requests # Necesario si aún quieres una ruta de proxy para el agente, pero lo integraremos directamente.

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder # <-- Importaciones clave para el prompt con memoria


# Flask-SocketIO (si aún lo necesitas para notificaciones externas, si no, puedes quitarlo)
from flask_socketio import SocketIO, emit

# --- Cargar variables de entorno al iniciar la aplicación ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# FLASK_API_URL ya no será necesario como una URL de API externa,
# sino que las llamadas al agente serán internas.

if not openai_api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada. Asegúrate de tener un archivo .env con OPENAI_API_KEY=tu_clave_aqui")

app = Flask(__name__)
# Configuración del puerto para Railway
port = int(os.getenv('PORT', 8080)) # Por defecto 8080 si PORT no está en env
socketio = SocketIO(app, cors_allowed_origins="*") # Mantener si usas las notificaciones de venta

# --- Configuración del Agente de LangChain (SIN RAG por ahora) ---
agent_llm = ChatOpenAI(model="GPT-4o-Mini", temperature=0.7, openai_api_key=openai_api_key)

# --- Definición de Herramientas (Tools) para el Agente ---

@tool
def AskAgent(query: str) -> str:
    """Proporciona información detallada sobre Glamping Brillo de Luna.
    Usa esta herramienta cuando el usuario pregunte sobre servicios, tipos de glamping,
    comodidades, políticas generales, o cualquier detalle sobre la experiencia del glamping.
    El agente puede usar esta herramienta para generar respuestas basadas en el conocimiento
    que se le ha dado indirectamente a través del prompt o si se le entrena con datos."""
    # Como NO estamos usando RAG por ahora, esta tool simplemente
    # devolverá una respuesta general o indicará que necesita más info.
    # Si quieres que el agente tenga "conocimiento" aquí sin RAG, tendrías que
    # codificar las respuestas directamente o usar un prompt muy bueno.
    # Por ejemplo, podríamos tener un pequeño diccionario de preguntas frecuentes aquí.
    if "servicios" in query.lower() or "ofrecen" in query.lower():
        return "Ofrecemos glampings de lujo con tinas de hidromasaje, cenas románticas, y actividades al aire libre como senderismo y observación de estrellas. Cada glamping tiene su propio baño privado y vistas espectaculares."
    elif "tipos de glamping" in query.lower() or "cuales glampings" in query.lower():
        return "Contamos con glampings tipo domo geodésico y burbujas transparentes. Cada uno ofrece una experiencia única de inmersión en la naturaleza con todas las comodidades."
    elif "precios" in query.lower() or "costo" in query.lower() or "tarifas" in query.lower():
        # Dirigimos al intent de Dialogflow si queremos que Dialogflow maneje la conversación.
        # O el agente podría decir algo más general y dejar que Dialogflow tome el control.
        return "Para información sobre precios y tarifas, te recomiendo visitar nuestra sección de 'Tarifas' en el menú principal o nuestro sitio web. Los precios varían según la temporada y el tipo de glamping."
    elif "reservas" in query.lower() or "reservar" in query.lower():
        # Igual, dirigir al intent de Dialogflow o dar una respuesta general.
        return "Puedes hacer una reserva directamente a través de nuestro sitio web en la sección de 'Reservas' o contactarnos directamente para asistencia personalizada."
    elif "ubicacion" in query.lower() or "lugar" in query.lower():
        return "Glamping Brillo de Luna está ubicado en un entorno natural privilegiado, a solo 2 horas de Medellín, ofreciendo una escapada perfecta de la ciudad. Te enviaremos la ubicación exacta al momento de tu reserva."
    else:
        return "Soy el agente de IA de Glamping Brillo de Luna. ¿Tienes alguna pregunta específica sobre nuestros glampings o servicios?"

# @tool
# def finalize_sale_process(product_info: str, client_info: str) -> str:
#     """Simula la finalización de un proceso de venta y emite una notificación.
#     Útil cuando el agente ha confirmado todos los detalles necesarios para registrar una venta."""
#     print(f"DEBUG: Venta finalizada para {client_info} de {product_info}. Enviando notificación...")

#     sale_data = {
#         "status": "OK",
#         "message": f"¡Nueva venta generada: {product_info} para {client_info}!",
#         "product": product_info,
#         "client": client_info,
#         "timestamp": datetime.now().isoformat()
#     }
#     socketio.emit('new_sale_notification', sale_data)
#     print("DEBUG: Notificación de venta emitida vía WebSocket.")

#     return f"¡Excelente! El proceso de venta de {product_info} para {client_info} ha sido exitoso y se ha generado la notificación."

tools = [
    # get_current_weather, # Considera si esta tool es relevante para Glamping Brillo de Luna
    AskAgent, # Nueva tool para info del glamping
    #finalize_sale_process
]

# --- Inicializar la memoria de conversación para el Agente LangChain ---
# Esta memoria se reiniciará con cada nueva solicitud de webhook a menos que
# la persistas externamente usando el session_id de Dialogflow.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Inicializar el Agente de LangChain ---
# Usamos ChatPromptTemplate para una mejor integración con el historial de mensajes
# y las funciones del agente.
prompt = PromptTemplate.from_template(
    """Actúa como un asistente útil y amigable de Glamping Brillo de Luna.
Tu objetivo es ayudar a los usuarios con información sobre el glamping, servicios, y responder a sus preguntas de manera conversacional.
Si el usuario pregunta sobre algo que no sabes, puedes sugerirle que revise las opciones del menú principal.

Herramientas disponibles:
{tools}

Usar las siguientes herramientas: {tool_names}

Tu historial de conversación:
{chat_history}

Pregunta actual del usuario: {input}

{agent_scratchpad}
"""
)
agent = create_react_agent(agent_llm, tools, prompt) # Ahora 'prompt' tiene todas las variables requeridas


# NOTA: Para un agente con memoria y tools, la estructura de prompt más robusta es con ChatPromptTemplate
# y MessagesPlaceholder. El `create_react_agent` ya espera un prompt específico.
# Si quieres usar un prompt más conversacional y estructurado, es más común hacer algo así:
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_core.messages import AIMessage, HumanMessage
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Actúa como un asistente amigable de Glamping Brillo de Luna."),
#         MessagesPlaceholder(variable_name="chat_history"), # Esto se llenará con la memoria
#         ("human", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"), # Esto es para las reflexiones del agente
#     ]
# )
# agent = create_openai_tools_agent(agent_llm, tools, prompt)
# Usaremos tu PromptTemplate simple por ahora, pero la advertencia de LangChain sugiere una migración.


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory, # ¡Aquí integramos la memoria!
    handle_parsing_errors=True
)

# --- Funciones Auxiliares para la Respuesta de Dialogflow ---
def build_response_json():
    return {
        "fulfillmentText": "",
        "fulfillmentMessages": [],
        "outputContexts": []
    }

def set_fulfillment_text(response, text_content):
    response["fulfillmentText"] = text_content

def add_quick_replies(response, text, replies):
    response["fulfillmentMessages"].append({
        "payload": {
            "facebook": { # Asumiendo que usas Facebook Messenger, si no, ajusta a "telegram" etc.
                "text": text,
                "quick_replies": replies
            }
        }
    })

def set_output_context(response, session, context_name, lifespan_count=5):
    response["outputContexts"].append({
        "name": f"{session}/contexts/{context_name}",
        "lifespanCount": lifespan_count
    })

def clear_output_context(response, session, context_name):
    response["outputContexts"].append({
        "name": f"{session}/contexts/{context_name}",
        "lifespanCount": 0
    })

# --- Webhook de Flask para Dialogflow ---
@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    print(f"Dialogflow Request Body: {json.dumps(req, indent=2)}")

    query_result = req.get('queryResult', {})
    user_query = query_result.get('queryText', '').strip()
    intent_display_name = query_result.get('intent', {}).get('displayName')
    parameters = query_result.get('parameters', {})
    session_id = req.get('session') # Obtener el ID de sesión de Dialogflow

    response_json = build_response_json() # Inicializa la respuesta de Dialogflow

    try:
        # --- Lógica de Manejo de Intents (prioridad sobre el agente si es explícito) ---
        if intent_display_name == 'Default Welcome Intent':
            print("Intent 'Default Welcome Intent' activado.")
            set_fulfillment_text(response_json, '¡Hola! Soy el asistente de Glamping Brillo de Luna. ¿En qué puedo ayudarte?')
            add_quick_replies(response_json, "¿Qué te gustaría saber?", [
                {"content_type": "text", "title": "Opciones Glamping", "payload": "GLAMPING_OPTIONS_PAYLOAD"},
                {"content_type": "text", "title": "Contactar Asesor", "payload": "CONTACT_ADVISOR_PAYLOAD"} # Ejemplo
            ])
            set_output_context(response_json, session_id, "main_menu_active")

        elif intent_display_name == 'Default Fallback Intent':
            # Si el intent de Fallback se activa. primero intenta con el agente
            print("Intent 'Default Fallback Intent' activado, intentando con Agente IA.")
            agent_output = ""
            try:
                # La memoria se usa aquí por cada invocación.
                # Si quieres persistencia entre turnos, necesitarías guardar/cargar
                # 'memory' usando 'session_id' en una DB externa.
                # Para un demo o pruebas simples, esta memoria funciona para una sola llamada.
                response_agent = agent_executor.invoke({"input": user_query})
                agent_output = response_agent.get("output", "Lo siento, no pude generar una respuesta con el agente.")
            except Exception as e:
                print(f"Error al invocar al agente LangChain en Fallback: {e}")
                agent_output = "Lo siento, el agente de IA no está disponible en este momento. Por favor, intenta de nuevo o selecciona una opción del menú."

            # Si el agente no pudo encontrar algo útil, o si devuelve "NO_INFO_ENCONTRADA_RAG"
            # puedes personalizar la respuesta o dirigir al menú.
            if "NO_INFO_ENCONTRADA_RAG" in agent_output: # Si tu tool AskAgent regresa esto.
                 set_fulfillment_text(response_json, f"Lo siento, no tengo información específica sobre '{user_query}'. ¿Hay algo más en lo que pueda ayudarte o te gustaría revisar las opciones del menú?")
            else:
                 set_fulfillment_text(response_json, agent_output)


        elif intent_display_name == 'Primer Menu':
            print("Intent 'Primer Menu' activado.")
            set_fulfillment_text(response_json, "¡Hola! ¿En qué puedo ayudarte hoy?")
            add_quick_replies(response_json, "¿Qué te gustaría hacer?", [
                {"content_type": "text", "title": "Opciones Glamping", "payload": "GLAMPING_OPTIONS_PAYLOAD"},
                {"content_type": "text", "title": "Más Información", "payload": "MORE_INFO_WEB_PAYLOAD"}
            ])
            set_output_context(response_json, session_id, "main_menu_active")

        elif intent_display_name == 'Glamping Options Menu':
            print("Intent 'Glamping Options Menu' activado.")
            set_fulfillment_text(response_json, "¿Sobre qué deseas saber de los glampings?")
            add_quick_replies(response_json, "Selecciona una opción:", [
                {"content_type": "text", "title": "Preguntar al Agente IA", "payload": "ASK_AI_AGENT_PAYLOAD"},
                {"content_type": "text", "title": "Reservas", "payload": "RESERVATIONS_PAYLOAD"},
                {"content_type": "text", "title": "Tarifas", "payload": "RATES_PAYLOAD"},
                {"content_type": "text", "title": "Ubicación", "payload": "LOCATION_PAYLOAD"}
            ])
            set_output_context(response_json, session_id, "glamping_options_menu_active")
            clear_output_context(response_json, session_id, "main_menu_active")

        elif intent_display_name == 'Ask AI Agent' or intent_display_name == 'langchainAgent':
            # Este intent (o si ya el usuario está en contexto de "preguntar al agente")
            # delega directamente al agente de LangChain
            print(f"Intent para Agente LangChain activado. Pregunta: \"{user_query}\"")
            
            agent_output = ""
            try:
                response_agent = agent_executor.invoke({"input": user_query})
                agent_output = response_agent.get("output", "Lo siento, no pude generar una respuesta con el agente.")
            except Exception as e:
                print(f"Error al invocar al agente LangChain: {e}")
                agent_output = "Lo siento, hubo un problema al procesar tu solicitud con el agente de IA. Por favor, intenta de nuevo."

            # Si el agente responde con "NO_INFO_ENCONTRADA_RAG", puedes personalizar la respuesta.
            if "NO_INFO_ENCONTRADA_RAG" in agent_output:
                set_fulfillment_text(response_json, f"No tengo esa información específica en este momento. ¿Hay algo más de Glamping Brillo de Luna en lo que pueda ayudarte?")
            else:
                set_fulfillment_text(response_json, agent_output)
            
            clear_output_context(response_json, session_id, "awaiting_ai_query") # Limpia el contexto si existe

        # elif intent_display_name == 'Confirmar_Venta_Intent':
        #     # Asumiendo que Dialogflow ya capturó los parámetros 'producto' y 'cliente'
        #     product = parameters.get('producto')
        #     client = parameters.get('cliente')
        #     if product and client:
        #         fulfillment_text_sale = finalize_sale_process(product, client)
        #         set_fulfillment_text(response_json, fulfillment_text_sale)
        #     else:
        #         set_fulfillment_text(response_json, "Necesito más información (producto y cliente) para finalizar la venta.")
        
        # --- Intents estáticos de ejemplo de Glamping Brillo de Luna ---
        elif intent_display_name == 'MenuOpcion_Horarios': # Renombrado de tu PruebaChatAI para consistencia
            set_fulfillment_text(response_json, "Nuestro horario de atención es de lunes a viernes de 8:00 AM a 5:00 PM y sábados de 9:00 AM a 1:00 PM. ¡Te esperamos en Glamping Brillo de Luna!")
        
        elif intent_display_name == 'MenuOpcion_Contacto':
            set_fulfillment_text(response_json, "Puedes contactarnos al 300-123-4567 o enviarnos un correo a info@brillodeluna.com. ¡Estamos para servirte!")
        
        elif intent_display_name == 'MenuOpcion_Soporte':
            set_fulfillment_text(response_json, "Para soporte técnico o cualquier incidencia durante tu estadía, por favor llama al 300-987-6543. ¡Estamos disponibles 24/7 para nuestros huéspedes!")
        
        elif intent_display_name == 'Reservas_Intent': # Puedes tener un intent específico de Dialogflow para esto
            set_fulfillment_text(response_json, "Puedes realizar tus reservas directamente en nuestro sitio web www.brillodeluna.com/reservas. ¡Es rápido y sencillo!")
        
        elif intent_display_name == 'Tarifas_Intent': # Un intent específico para tarifas
            set_fulfillment_text(response_json, "Nuestras tarifas varían según la temporada y el tipo de glamping. Visita www.brillodeluna.com/tarifas para ver todos los detalles y ofertas actuales.")
        
        elif intent_display_name == 'Ubicacion_Intent': # Un intent para la ubicación
            set_fulfillment_text(response_json, "Glamping Brillo de Luna se encuentra en un hermoso paraje natural cerca de [Nombre Ciudad Cercana]. Te enviaremos la ubicación exacta y cómo llegar con tu confirmación de reserva.")
        
        else:
            # Si ningún intent específico coincide, el Agente IA podría intentar responder
            print(f"DEBUG: Intent '{intent_display_name}' no manejado directamente. Intentando con Agente IA...")
            agent_output = ""
            try:
                response_agent = agent_executor.invoke({"input": user_query})
                agent_output = response_agent.get("output", "Lo siento, no pude generar una respuesta con el agente.")
            except Exception as e:
                print(f"Error al invocar al agente LangChain para intent no manejado: {e}")
                agent_output = "Lo siento, hubo un problema con el asistente. ¿Podrías intentar de otra forma o seleccionar una opción del menú?"
            
            set_fulfillment_text(response_json, agent_output)


    except openai.APIError as e:
        print(f"Error de API de OpenAI: {e}")
        set_fulfillment_text(response_json, "Lo siento, hubo un problema con el servicio de IA. Por favor, inténtalo más tarde.")
    except Exception as e:
        print(f"Error general en el procesamiento del webhook: {e}")
        set_fulfillment_text(response_json, f"Lo siento, algo salió mal con el asistente. Error: {e}. Por favor, inténtalo de nuevo.")

    return jsonify(response_json)

# --- WebSocket Event Handlers (mantener si se usan) ---
@socketio.on('connect')
def test_connect():
    print('Cliente WebSocket conectado!')

@socketio.on('disconnect')
def test_disconnect():
    print('Cliente WebSocket desconectado.')

if __name__ == '__main__':
    print("Iniciando Flask con SocketIO...")
    # Asegúrate de que Railway usa el puerto 8080 si esa es tu configuración de servicio.
    # En Railway, si no especificas nada, usa el puerto expuesto por defecto (a menudo 8080).
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)