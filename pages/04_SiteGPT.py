import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="Site GPT",
    page_icon="🌍",
)
st.title("Site GPT")

st.markdown("searching data in website.")

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second =1
    docs = loader.load()
    st.write(docs)
    

html2Text_transformer = Html2TextTransformer()

with st.sidebar:
    url = st.text_input("Write down a Url", placeholder="https://example.com")


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("please write down a Sitemap url.")
    else:
        load_website()
