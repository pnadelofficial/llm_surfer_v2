import streamlit as st

def searcher_cb(i, length, pbar):
    pbar.progress(((i/length) + (1/length)), text='Collecting relevant documents...')
    if i == length - 1:
        pbar.empty()
        st.success("Collection complete!")  

def embedder_cb(i, length, pbar):
    pbar.progress(((i/length) + (1/length)), text='Embedding documents...')
    if i == length - 1:
        pbar.empty()
        st.success("Embedding complete!")

def surfer_cb(i, length, result, out): # , pbar):
    # pbar.progress(((i/length) + (1/length)), text='Determining relevancy...')
    st.markdown(f"Result: **{result['alt_title']}**")
    st.markdown(f"Relevancy: **{out['relevancy']}**")
    st.markdown(f"Comment: *{out['comment']}*")
    add_info = [k + ': ' + v for k, v in out.items() if k not in ['title', 'url', 'relevancy', 'comment']]
    for info in add_info:
        st.markdown(info)
    st.divider()
    
    # if i == length - 1:
    #     pbar.empty()
    #     st.success("Relevancy determination complete!")
    