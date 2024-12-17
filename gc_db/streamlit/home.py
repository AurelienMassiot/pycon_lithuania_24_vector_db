import logging
import sys

import streamlit as st

from gc_db.config import LOGO_PATH
from gc_db.streamlit.init_db_st import init_streamlit
from gc_db.streamlit.st_creators import image_as_query, set_style_css
from gc_db.streamlit.st_utils import perform_query, translate_query
from gc_db.utils.utils import style_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "--hnsw" in sys.argv:
    hnsw = True
    st.session_state["hnsw"] = hnsw
else:
    hnsw = False
    st.session_state["hnsw"] = hnsw

st.session_state["k"] = 20

VDB_IM, FCLIP, SEG = init_streamlit(hnsw)

st.set_page_config(layout="wide")
st.title("MULTI-MODAL SEARCH ENGINE")

set_style_css(30, 25)

with st.sidebar:
    st.image(LOGO_PATH, width=200)
    query_text = st.text_input(label="**Your request:**")
    if hasattr(VDB_IM, 'query_with_kmeans'):
        st.checkbox("**Use inverted file index**", key="use_ivf")
        if st.session_state["use_ivf"]:
            st.number_input("**Number of probes:**", key="n_probes", value=1, step=1)

        if st.session_state["use_ivf"]:
            st.number_input("**Number of clusters:**", key="n_clusters", value=10, step=1)
            reindex = st.button("Re-index")
            if reindex:
                VDB_IM.inverted_index = {}
                VDB_IM.init_kmeans_index(nb_clusters=st.session_state["n_clusters"])
                st.write(len(VDB_IM.inverted_index))

tab1, tab2 = st.tabs(["Text", "Image"])
use_ivf = st.session_state["use_ivf"] if "use_ivf" in st.session_state else False
with tab1:
    if query_text != "":
        logger.info(f"Performing query with {hnsw}" + str(st.session_state["VDB_NMS"]))
        logger.info(f"Request {query_text} ")
        translated_query_text = translate_query(query_text)
        logger.info(f"Translated {query_text} to {translated_query_text}")

        if hnsw:
            logger.info(f"Performing query with {hnsw}" + str(st.session_state["VDB_NMS"]))
            perform_query(st.session_state["VDB_NMS"], FCLIP, translated_query_text, use_kmeans_query=use_ivf)
        else:
            perform_query(VDB_IM, FCLIP, translated_query_text, use_kmeans_query=use_ivf)
        st.table(style_df(st.session_state["log_dataframe"], 30))

with tab2:
    uploaded_file = st.file_uploader("Choose an image to upload", accept_multiple_files=False, )
    if uploaded_file is not None:
        cloth = image_as_query(uploaded_file)
        logger.info(f"hnsw flag{hnsw}")
        perform_query(VDB_IM, FCLIP, cloth, use_kmeans_query=use_ivf)
