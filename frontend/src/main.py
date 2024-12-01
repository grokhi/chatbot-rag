import os

import gradio as gr
import requests

API_BASE_URL = "http://{}:{}".format(os.getenv("HOST", "localhost"), os.getenv("PORT", 8000))


def query_chatbot(chat_history, user_message):
    """
    Send a query to the backend and return the updated chat history.

    Args:
        chat_history (list): Current chat history as a list of messages.
        user_message (str): User's message to the chatbot.

    Returns:
        list: Updated chat history with the bot's response.
    """
    # Ensure the chat_history is a list of dictionaries
    if not chat_history:
        chat_history = []

    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_message})

    # Prepare the payload and send the query to the backend
    payload = {"query": user_message, "context": {}}
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            bot_response = (
                f"{data['generation']}\n\n"
                f"**Web Search Info:** {data['web_search']}\n\n"
                f"**Documents:**\n" + "\n".join([f"- {doc}" for doc in data.get("documents", [])])
            )
            chat_history.append({"role": "assistant", "content": bot_response})
        else:
            error_message = f"Error: {response.status_code}\n{response.text}"
            chat_history.append({"role": "assistant", "content": error_message})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Return updated chat history and clear user input
    return chat_history, ""


# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 🤖 RAG Chatbot")

    with gr.Column():
        chat_display = gr.Chatbot(label="Chat with RAG Bot", type="messages")
        user_input = gr.Textbox(label="", placeholder="Type your message here...")
        send_button = gr.Button("Send")

    # Event to update the chat history
    send_button.click(
        query_chatbot,
        inputs=[chat_display, user_input],
        outputs=[chat_display, user_input],
        queue=False,
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)
