import os
import time
import uuid

import openai
import streamlit as st
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.llms import OpenAI

from frontend.pages.ui_backend import interface
from src.tools.pdf import open_pdf


def main():
    interface.init_page("Chat with PDF", icon="💬")
    check_api_key()

    files = interface.upload_pdf_file(multiple=True, on_change=clear_chat)

    if files:
        init_first_message()

        if prompt := st.chat_input(
            "Your question"
        ):  # Prompt for user input and save to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            index = index_data(files, st.session_state.user_id)
            chat_engine = index.as_chat_engine(
                chat_mode="condense_question", verbose=True
            )

        for (
            message
        ) in st.session_state.messages:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {
                        "role": "assistant",
                        "content": response.response,
                    }
                    st.session_state.messages.append(
                        message
                    )  # Add response to message history

    else:
        st.info("Please upload a PDF file.")


def delete_files(user_id, dir_path="./data"):
    save_path = os.path.join(dir_path, user_id)

    if os.path.exists(save_path):
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            os.remove(file_path)


def clear_chat():
    if "messages" in st.session_state:
        del st.session_state.messages


def check_api_key():
    if not st.session_state.get("api_key"):
        api_key = st.text_input(
            "Enter your API key",
            help="Your API key won't be stored",
            type="password",
        )

        if api_key:
            if api_key == st.secrets.admin_pass:
                api_key = st.secrets.openai_key

            make_test_call(api_key)

        st.stop()
    else:
        openai.api_key = st.session_state.api_key


def make_test_call(api_key):
    # Initialize the OpenAI API client
    openai.api_key = api_key

    # Test API call to check if the API key is correct
    try:
        # Replace with a valid API call (e.g., completion, classification, etc.)
        _ = openai.Completion.create(
            engine="davinci-002", prompt="Once upon a time,", max_tokens=50
        )

        # Print the response if successful
        st.success("API key is correct!")
        st.session_state.api_key = api_key
        st.session_state.user_id = str(uuid.uuid4())

        time.sleep(1)
        st.experimental_rerun()

    except openai.error.OpenAIError as e:
        # If there's an error, print the error message
        st.error(f"{str(e)}")


def save_files_locally(files, user_id, dir_path):
    save_path = os.path.join(dir_path, user_id)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in files:
        doc = open_pdf(file)
        doc.ez_save(f"{save_path}/{file.name}")


def init_first_message():
    if (
        "messages" not in st.session_state
    ):  # Initialize the chat message history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about the uploaded PDF file",
            }
        ]


@st.cache_resource(show_spinner=False)
def index_data(files, user_id, dir_path="./data"):
    with st.spinner("Saving data..."):
        save_files_locally(files, user_id, dir_path)

    with st.spinner(
        text="Loading and indexing your data! This should take 1-2 minutes."
    ):
        reader = SimpleDirectoryReader(input_dir=dir_path, recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                system_prompt="""
                You are an expert on the documents uploaded by the user. \
                Your role is to respond to any inquiries the user has regarding these documents. \
                Assume all questions are directly related to the uploaded content. \
                If the user seeks the source of a particular claim, display relevant \
                text excerpts from the uploaded document containing the pertinent \
                information. Maintain brevity in your responses, unless the user \
                requests a comprehensive explanation. Ensure all claims are factual \
                and strictly based on the information available within the \
                documents; avoid inventing any facts not present in the provided content.
                """,
            )
        )
        index = VectorStoreIndex.from_documents(
            docs, service_context=service_context
        )

        delete_files(user_id, dir_path)

        return index


if __name__ == "__main__":
    main()
