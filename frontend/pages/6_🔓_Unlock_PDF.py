import streamlit as st
from frontend.pages.ui_backend import interface
from src.tools.pdf import open_pdf
import io


def main():
    interface.init_page('Unlock PDF file', icon='🔓')
    file = interface.upload_pdf_file()

    if file:
        fname = file.name.rstrip('.pdf')

        with st.spinner("Loading file..."):
            pdf = open_pdf(file)
        
        unlocked_pdf = unlock_pdf(pdf)
        if unlocked_pdf:
            unlocked_file = io.BytesIO()
            unlocked_pdf.ez_save(unlocked_file, linear=True)

            st.download_button(
            "Download Unlocked PDF",
            unlocked_file,
            file_name=f"{fname}_unlocked",
            mime='application/pdf'
        )

        

def unlock_pdf(pdf):
    if not pdf.needs_pass:
        st.success("File is not password protected")
        return

    password = st.text_input(label="Enter password: ", type='password')
    
    if password:
        unlocked_file = unlock_with_pass(pdf, password)
    
        return unlocked_file


    
def unlock_with_pass(pdf, password):
    if pdf.authenticate("") == 2:
        st.success(
            "The document is protected by an owner, "
            "but not by a user password. No further action needed."
        )
        return pdf

    return check_password(pdf, password)
    
    
def check_password(pdf, password):   
    rc = pdf.authenticate(password)

    if rc == 2:
        st.success("Sucessfully authenticated with user password.")
        return pdf
    
    if rc in (4, 6):
        st.success(
            "Sucessfully authenticated with owner password."
        )
        return pdf
    
    if rc > 0:
        st.sucess("Sucessfully authenticated.")
        return pdf
    
    st.error("Wrong password. Unable to authenticate.")

if __name__ == "__main__":
    main()
