import streamlit as st
from frontend.pages.ui_backend import interface
from src.tools.pdf import open_pdf
import io
import fitz

perms = {
    fitz.PDF_PERM_PRINT: "Print document",
    fitz.PDF_PERM_COPY: "Copy text and graphics",
    fitz.PDF_PERM_ANNOTATE: "Annotate text and fields",
    fitz.PDF_PERM_MODIFY: "Modify contents",
    fitz.PDF_PERM_FORM: "Fill forms and sign",
    fitz.PDF_PERM_ASSEMBLE: "Insert, rotate, or delete pages",
    fitz.PDF_PERM_PRINT_HQ: "High quality print",
}


def main():
    interface.init_page('Lock PDF file', icon= '🔐')
    file = interface.upload_pdf_file()

    if file:
        fname = file.name.rstrip('.pdf')

        with st.spinner("Loading file..."):
            pdf = open_pdf(file)
        
        col1, col2 = st.columns(2)

        with col1:
            user_pass = st.text_input("User password: ", type="password")
            owner_pass = st.text_input("Owner password: ", type="password")

        with col2:
            selected_perms_lst = st.multiselect(
                "Select user permissions: ",
                default = list(perms.keys())[:3],
                options=perms.keys(),
                format_func=lambda x: perms[x],
            )
        
        lock_action = st.button('Lock PDF')
        if lock_action:
            locked_file = lock_pdf(pdf, selected_perms_lst, user_pass, owner_pass)

            st.download_button(
                "Download Locked PDF",
                locked_file,
                file_name=f"{fname}_locked",
                mime='application/pdf'
            )

def lock_pdf(pdf, perms_lst, user_pass, owner_pass):
    if perms_lst:
        # Initialize the result with the first permission
        selected_perms = perms_lst[0]

        # Iterate through the list starting from the second permission
        for permission in perms_lst[1:]:
            selected_perms |= permission
    else:
        selected_perms = fitz.PDF_PERM_ACCESSIBILITY
    

    locked_file = io.BytesIO()
    pdf.save(
        locked_file,
        garbage=3,
        deflate=True,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        permissions=selected_perms,
        user_pw=user_pass,
        owner_pw=owner_pass
    )

    return locked_file

if __name__ == "__main__":
    main()
