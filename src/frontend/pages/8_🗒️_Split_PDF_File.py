import io

import streamlit as st

from src.frontend.pages.ui_backend import interface
from src.tools.pdf import open_pdf, split_doc


def main():
    interface.init_page("Split PDF file", icon="🗒️")

    file = interface.upload_pdf_file()

    if file:
        pdf = open_pdf(file)
        fname = file.name.rstrip(".pdf")

        col1, col2 = st.columns(2)
        with col1:
            st.write(fname)
            split = st.button("Split file")

        with col2:
            pages = st.multiselect("On pages", range(1, len(pdf)))

        if split:
            split_pdf(pdf, pages, fname)


def split_pdf(pdf, pages, fname):
    if len(pages) < 1:
        st.error("Select at least one page to split on")
        st.stop()

    with st.spinner("Splitting PDF..."):
        split_pdfs = split_doc(pdf, split_on_pages=sorted(pages))
        st.success("PDF split successfully!")

    for i, pdf in enumerate(split_pdfs):
        file = io.BytesIO()
        pdf.ez_save(file, linear=True)

        st.download_button(
            f"Download Split {i + 1}",
            file,
            file_name=f"{fname}_part{i + 1}",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
