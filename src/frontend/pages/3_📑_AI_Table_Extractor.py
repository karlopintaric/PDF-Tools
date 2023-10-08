import time

import cv2
import numpy as np
import streamlit as st
from mitosheet.streamlit.v1 import spreadsheet

from src.frontend.ui_backend import interface
from src.extraction.extractors import TableDataExtractor, TableExtractor
from src.objects.documents import PdfDocument


# Main function
def main():
    interface.init_page(title="PDF Table Extractor", icon="📑")
    table_extractor, data_extractor = init_extractors()

    # Upload a PDF file
    file = interface.upload_pdf_file(on_change=new_file)

    if "new" not in st.session_state:
        st.session_state.new = True

    if file:
        display_tables(file, table_extractor, data_extractor)

    else:
        st.info("Please upload a PDF file.")


# Function to create table extractors
@st.cache_resource(show_spinner="Loading models...")
def init_extractors():
    return TableExtractor(), TableDataExtractor()


def new_file():
    interface.clear_state()
    st.session_state.new = True


# Function to extract tables from the PDF
def extract_tables(file, extractor):
    if "data_tables" in st.session_state:
        return [table.image for table in st.session_state.data_tables.values()]

    if "tables" in st.session_state:
        return st.session_state.tables

    pdf = PdfDocument(file)
    st.session_state.tables = []

    return extractor.extract(pdf)


def extract_data_from_tables(tables, extractor):
    if "data_tables" in st.session_state:
        return st.session_state.data_tables.values()

    st.session_state.data_tables = {}

    return extractor.extract(tables)


def reduce_visibility(img):
    # Define the opacity level (alpha) as a value between 0 and 1
    opacity = 0.5  # You can adjust this value to control the opacity
    # Create a transparent image with the same size as the original image
    transparent_layer = np.zeros_like(img, dtype=np.uint8)

    # Blend the original image with the transparent layer using alpha blending
    return cv2.addWeighted(img, 1 - opacity, transparent_layer, opacity, 0)


def edit_spreadsheets():
    dfs, _ = spreadsheet(
        *[table.df for table in st.session_state.data_tables.values()],
        df_names=list(st.session_state.data_tables.keys()),
    )

    for table_name in dfs:
        st.session_state.data_tables[table_name].df = dfs[table_name]


def display_tables(file, table_extractor, data_extractor):
    container = st.container()
    data_view = container.toggle("Extract data", disabled=st.session_state.new)

    # Extract tables from the uploaded PDF
    tables = extract_tables(file, table_extractor)

    if not data_view:
        # Display tables
        display_tables_images(tables)
        if st.session_state.new:
            found_tables = len(st.session_state.tables)
            container.success("Finished!")
            container.write(f"Found {found_tables} tables")
            time.sleep(1)

            if found_tables:
                st.session_state.new = False
                st.experimental_rerun()

    else:
        with st.spinner("Extracting data from tables..."):
            tables_data = extract_data_from_tables(tables, data_extractor)
            display_tables_data(tables_data)

        with container:
            edit_spreadsheets()


# Function to display tables and convert to data_view
def display_tables_images(tables):
    with st.spinner("Loading tables..."):
        # Display each table image
        columns = st.columns(3, gap="medium")
        containers = [col.container() for col in columns]

        for i, table in enumerate(tables):
            img = table.img
            fname = table.fname
            container = containers[i % 3]

            if "new" not in st.session_state or st.session_state.new:
                st.session_state.tables.append(table)
                img = reduce_visibility(img)

            with container:
                # Display the table image
                st.image(
                    img,
                    use_column_width=True,
                    caption=f"Page {table.page_num + 1}, Table {table.table_num + 1}",
                )

                # Add download button
                interface.download_image(
                    img,
                    label="Download table",
                    fname=fname,
                    disabled=st.session_state.new,
                )

                if st.button("Remove table", key=f"remove_{fname}"):
                    if "data_tables" in st.session_state:
                        st.session_state.data_tables.pop(fname, None)

                    else:
                        st.session_state.tables.pop(i)

                    st.experimental_rerun()


def display_tables_data(tables):
    for table in tables:
        st.text(" ")
        container = st.container()

        markdown_table = table.markdown
        fname = table.image.fname

        if fname not in st.session_state.data_tables:
            st.session_state.data_tables[fname] = table

        with container:
            st.text(
                f"Table {table.image.table_num + 1} from page {table.image.page_num + 1}"
            )
            st.markdown(markdown_table, unsafe_allow_html=False)
            # Add a download button for the data_view
            st.download_button(
                label="Download as CSV",
                data=table.df.to_csv().encode("utf-8"),
                file_name=f"{fname}_data",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
