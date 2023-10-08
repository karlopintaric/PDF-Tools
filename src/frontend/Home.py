import streamlit as st

st.set_page_config(page_title="Homepage", page_icon="🏠", layout="wide")

# Title and description
st.title("📚 PDF Toolbox App Collection")
st.write("Welcome to the PDF Toolbox App Collection! Explore a variety of PDF tools designed to streamline your PDF management tasks. 🛠️")

# App Options
st.header("🛠️ Available PDF Apps")

# Define the available apps and their respective names
apps = {
    "1_🖼️_PDF_to_Image.py": "PDF to Image Converter",
    "2_🎨_Images_to_PDF.py": "Images to PDF Converter",
    "3_📑_AI_Table_Extractor.py": "AI Table Extractor",
    "4_💬_AI_Chat_with_PDF.py": "AI Chat with PDF",
    "5_🗂️_Merge_PDF_Files.py": "PDF Merger",
    "6_🔓_Unlock_PDF.py": "PDF Unlocker",
    "7_🔐_Lock_PDF.py": "PDF Locker",
    "8_🗒️_Split_PDF_File.py": "PDF Splitter",
    "9_🧽_Clean_PDF.py": "PDF Cleaner"
}

# Display app options
for app_key, app_name in apps.items():
    st.write(f"- {app_name}")

# Get Started
st.header("🚀 Get Started")
st.write("Select an app from the sidebar to get started with PDF management. "
         "Each app is designed to handle specific tasks related to PDFs. "
         "Happy PDF managing! 😊")

# Add author and GitHub link
st.subheader("Author")
st.write("Karlo Pintarić")

st.subheader("GitHub Repository")
st.write("[PDF-Tools GitHub Repo](https://github.com/karlopintaric/PDF-Tools)")

st.sidebar.success("Select a tool")
