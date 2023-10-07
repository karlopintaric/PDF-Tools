import streamlit as st
import cv2


def init_page(title, icon):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.title(f"{title} {icon}")


def upload_pdf_file(multiple=False, **kwargs):
    if multiple:
        label = "Upload PDF files"
    else:
        label = "Upload a PDF file"

    uploaded_file = st.file_uploader(
        label, type=["pdf"], accept_multiple_files=multiple, **kwargs
    )
    return uploaded_file


def clear_state():
    for key in st.session_state.keys():
        del st.session_state[key]


def download_image(img, label, fname, disabled=False):
    st.download_button(
        label=label,
        data=cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[
            1
        ].tobytes(),
        key=fname,
        file_name=fname,
        mime="image/png",
        use_container_width=True,
        disabled=disabled,
    )
