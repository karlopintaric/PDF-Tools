import streamlit as st
from src.tools.pdf import open_pdf
import io
from frontend.pages.ui_backend import interface


def main():
    interface.init_page('Clean PDF file', icon='🧽')
    st.write(
        """Removes potentially malicious code from PDF files, \
        resets form fields and reduces file size if possible"""
    )

    # Upload a PDF file
    file = interface.upload_pdf_file()

    if file:
        fname = file.name.rstrip('.pdf')

        st.write(fname)
        sanitize_action = st.button("Clean file")

        if sanitize_action:
            cleaned_file = clean_pdf(file)

            st.download_button(
            "Download cleaned PDF",
            cleaned_file,
            file_name=f"{fname}_cleaned",
            mime='application/pdf'
        )
            

def clean_pdf(file):
    with st.spinner("Cleaning PDF..."):
        pdf = open_pdf(file)

        pdf.scrub()
        cleaned_file = io.BytesIO()
        pdf.save(cleaned_file, garbage=4, deflate=True, clean=True, linear=True)
        st.success("PDF cleaned successfully!")

    return cleaned_file


if __name__ == "__main__":
    main()
