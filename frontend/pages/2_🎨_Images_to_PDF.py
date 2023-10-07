import streamlit as st
from src.objects.documents import PdfDocument
from frontend.pages.ui_backend import interface
import io


def main():
    interface.init_page(title="Images to PDF", icon="🎨")
    uploaded_files = st.file_uploader(
        "Upload images", type=["png"], accept_multiple_files=True
    )
    
    if uploaded_files:
        convert = st.button('Convert')
        container = st.container()
        
        cols = st.columns(3)
        for i, image in enumerate(uploaded_files):
            col = cols[i % 3]

            with col:
                st.image(image, caption=image.name)
            
        if convert:
            pdf_file = convert_images_to_pdf(uploaded_files)
            
            with container:
                st.download_button(
                    "Download Converted PDF",
                    pdf_file,
                    file_name='images',
                    mime="application/pdf",
                )


def convert_images_to_pdf(uploaded_files):
    with st.spinner('Converting to PDF'):
        pdf = PdfDocument.from_images(uploaded_files)
        
        file = io.BytesIO()
        pdf.doc.ez_save(file, linear=True)
    
    return file

if __name__ == "__main__":
    main()
