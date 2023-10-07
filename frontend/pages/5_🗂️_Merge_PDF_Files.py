import streamlit as st
from src.tools.pdf import merge_docs
import io
from frontend.pages.ui_backend import interface


def main():
    interface.init_page("Merge PDF files", icon="🗂️")

    files = interface.upload_pdf_file(multiple=True)
    merge = st.button("Merge files")

    fnames = []
    for file in files:
        fname = file.name.rstrip(".pdf")

        st.write(fname)
        fnames.append(fname)

    if merge:
        merged_file = merge_pdfs(files)
        merged_fname = "_".join(fnames)

        st.download_button(
            "Download Merged PDF",
            merged_file,
            file_name=f"{merged_fname}_merged",
            mime="application/pdf",
        )


def merge_pdfs(files):
    if not len(files) > 1:
        st.error("Need to upload at least two files to merge")
        st.stop()

    with st.spinner("Merging PDFs..."):
        merged_file = io.BytesIO()
        merged_pdf = merge_docs([pdf for pdf in files])
        merged_pdf.ez_save(merged_file, linear=True)

        st.success("PDFs merged successfully!")

    return merged_file


if __name__ == "__main__":
    main()
