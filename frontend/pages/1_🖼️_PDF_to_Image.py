import streamlit as st

from frontend.pages.ui_backend import interface
from src.objects.documents import PdfDocument


def main():
    interface.init_page(title="PDF to Image", icon="🖼️")
    uploaded_file = interface.upload_pdf_file()

    if uploaded_file:
        load_pdf_pages(uploaded_file)
        display_image_and_download()
    else:
        interface.clear_state()


def load_pdf_pages(uploaded_file):
    pdf = PdfDocument(uploaded_file)

    # Update session state
    if "pages" not in st.session_state:
        with st.spinner("Loading pages"):
            pages = list(pdf.page_images)
            st.session_state.pages = pages


def display_image_and_download():
    col, _ = st.columns([1, 10])
    i = col.number_input(
        "Page: ",
        min_value=1,
        value=1,
        max_value=len(st.session_state.pages),
    )

    page_image = st.session_state.pages[i - 1]
    fname = page_image.fname
    img = page_image.img

    st.image(img)

    interface.download_image(
        img, label="Download page as PNG file", fname=fname
    )


if __name__ == "__main__":
    main()
