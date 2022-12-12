import streamlit as st
import pandas as pd
import numpy as np
import uuid
from dnn import CNNModel, train, HFDataset, predict_docs_batch, get_embedder, get_tokenizer
from utils import split_data, load_df_data, comput_max_stc_len, expand_dict, read_file, create_folder
from texpl import texpl_scores_all, texpl_peakdet_process, cluster_embeddings, embed_corpus
from texpl import peak_det_sensit, clustering_sensit
from plotting import plot_train_loss, plot_num_phrases, plot_num_clusters, cluster_visualization
import streamlit.components.v1 as components

state = st.session_state
st.set_page_config(page_title="Text narratives analyzer", layout="wide")

st.title('Text narratives analyzer')

def init_state(key, value):
    if key not in state:
        state[key] = value

init_state('show_params', False)

for s in ['model', 'docs', 'labels', 'uq_ids', 'res_scores', 'res_explan', 'res_clusters', 'corpus', 'embeddings']:
    init_state(s, None)

if 'embedder' not in state:
    state['embedder'] = get_embedder()
    
if 'tokenizer' not in state:
    state['tokenizer'] = get_tokenizer()

if 'sess_id' not in state:
    state['sess_id'] = str(uuid.uuid4())[:8]
    
create_folder("tmp")

@st.cache
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


def run_training(df, narrative_column, output_column, training_epochs, ids_col):
    docs, labels, uq_ids = load_df_data(df, narrative_column, output_column, ids_col=ids_col, min_stc_len=20)
    state.docs, state.labels, state.uq_ids = docs, labels, uq_ids
    train_docs, test_docs, train_labels, test_labels = split_data(docs, labels)
    max_stc_len = comput_max_stc_len(docs)
    state.tokenizer.model_max_length = max_stc_len
    model = CNNModel(vocab_size=state.tokenizer.vocab_size, drop_proba=.5, num_filters=512)
    train_dataset = HFDataset(state.tokenizer(train_docs, padding=True, truncation=True), train_labels)
    test_dataset = HFDataset(state.tokenizer(test_docs, padding=True, truncation=True), test_labels)
    training_progres_bar = st.progress(0)
    update_pg_bar = lambda n, _ : training_progres_bar.progress((n + 1) / training_epochs)
    train_res = train(model, train_dataset, test_dataset, n_epochs=training_epochs, epoch_callback=update_pg_bar) 
    plot_train_loss(train_res.summary_df(), out_file=f"tmp/epochs_plot_{state.sess_id}.png", dpi=84)
    return model
   
def identify_phrases():
    predict_fn = lambda x: predict_docs_batch(x, state.model, state.tokenizer, batch_size=64, hf_model=False)
    samples_idx = np.where(state.labels == 1)[0]
    
    gen_progres_bar = st.progress(0)
    update_pg_bar = lambda n : gen_progres_bar.progress(n / len(samples_idx))
    state.res_scores = texpl_scores_all(samples_idx, state.docs, ws=6, predict_fn=predict_fn, progress_cb=update_pg_bar)

def cluster_phrases(sd_threshold=1.0, distance_threshold=1.25):
    state.res_explan = texpl_peakdet_process(state.res_scores, state.docs, state.uq_ids, sd_threshold=sd_threshold)
    state.corpus = expand_dict(state.res_explan, 'text')
    state.embeddings = embed_corpus(state.corpus, state.embedder)
    state.res_clusters = cluster_embeddings(state.embeddings, distance_threshold=distance_threshold)
    

def validations(df, narrative_col, output_col):
    if narrative_col == '' or output_col == '':
        st.error("The narrative and output columns can not be empty")
        st.stop()
    labels = df[output_col].values
    if not set(1*np.unique(labels[~np.isnan(labels)])).issubset({0, 1}):
        st.error("The target column must be binary (contain only zeros and ones)")
        st.stop()
    if not all([isinstance(i, str) for i in df[narrative_col].values]):
        st.error("The narrative column must containt only text")
        st.stop()

st.write("### Upload data")
uploaded_file = st.file_uploader("Upload your crash narratives in CSV format")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df_cols = list(df.columns.values)
    st.write("### Select analysis parameters")
    with st.form("training_form"):
        lay_col1, lay_col2, lay_col3, lay_col4, _ = st.columns([1, 1, 1, 1, 4])
        with lay_col1:
            narrative_col = st.selectbox('Narrative column *', [''] + df_cols,
                                         help="Column that contains the text narrative")
        
        with lay_col2:
            output_col = st.selectbox('Target column *', [''] + df_cols,
                                      help="Column that contains the binary output variable (1s and 0s)")
        
        with lay_col3:
            training_epochs = st.number_input('Training Epochs', value=10, help="Number of epochs for model training")
        
        with lay_col4:
            ids_col = st.selectbox('Unique IDs column', [''] + df_cols,
                                         help="Column that contains the unique IDs")

        st.caption("(*) Required")
        btn_run_training = st.form_submit_button("Run Training")
        if btn_run_training:
            validations(df, narrative_col, output_col)
            state.model = run_training(df, narrative_col, output_col,
                                       training_epochs, ids_col if ids_col != '' else None)
            state.res_explan, state.res_clusters, state.res_scores = None, None, None  # Disable further stages


if state.model is not None:  # Training is complete
    st.image(f"tmp/epochs_plot_{state.sess_id}.png", output_format='PNG') 
    with st.expander("Interpreting loss curves"):
        st.image("files/img/loss_interpret.png", output_format='PNG')

    with st.form("identif_phrases"):  
        btn_identify_phrases = st.form_submit_button("Identify Correlated Phrases")
        if btn_identify_phrases:
            identify_phrases()
            with st.spinner(text="Clustering phrases"):
                cluster_phrases(sd_threshold=1.0, distance_threshold=1.25)

if state.res_scores is not None:  # scores generation complete
    if st.button("Update generation parameters", disabled=state.show_params):
        state.show_params = True
        with st.spinner(text="Conducting sensitivity analysis"):
            thres_vals, n_phrases = peak_det_sensit(state.res_scores)
            plot_num_phrases(thres_vals, n_phrases, out_file=f"tmp/n_phrases_plot_{state.sess_id}.png", dpi=84)
            dis_thres, n_clusters = clustering_sensit(state.embeddings)
            plot_num_clusters(dis_thres, n_clusters, out_file=f"tmp/n_clusters_plot_{state.sess_id}.png", dpi=84)
            

    if state.show_params:
        with st.form("expl_form"):
            lay_col1, lay_col2, lay_col3 = st.columns([2, 2, 4])
    
            with lay_col1:
                sd_threshold = st.slider('Peak Detection Threshold', value=1.0, min_value=.75, max_value=2.0, step=.25)
                st.image(f"tmp/n_phrases_plot_{state.sess_id}.png")

            with lay_col2:
                dist_threshold = st.slider('Clustering Threshold', value=1.25, min_value=.75, max_value=1.5, step=.25)
                st.image(f"tmp/n_clusters_plot_{state.sess_id}.png")  

            if st.form_submit_button("Update"):
                with st.spinner(text="Updating"):
                    cluster_phrases(sd_threshold=sd_threshold, distance_threshold=dist_threshold)
                    dis_thres, n_clusters = clustering_sensit(state.embeddings)
                    plot_num_clusters(dis_thres, n_clusters, out_file=f"tmp/n_clusters_plot_{state.sess_id}.png", dpi=84)
                    st.experimental_rerun()

if state.res_clusters is not None:
    cluster_visualization(state.res_clusters, state.corpus, expand_dict(state.res_explan, 'scores'),
                          expand_dict(state.res_explan, 'uq_ids'), template_file=f"files/img/bar_template.html",
                          out_file=f"tmp/bar_{state.sess_id}.html")
    components.html(read_file(
        f"tmp/bar_{state.sess_id}.html"
    ), width=1300, height=2000, scrolling=True)
    with open(
        f"tmp/bar_{state.sess_id}.html"
        , "rb") as f:
        st.download_button(label="Download", data=f, file_name=f'correlated_phrases_{state.sess_id}.html',
                           mime='text/html')
    

