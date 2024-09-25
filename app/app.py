import streamlit as st
from openai import OpenAI
from default_prompts import research_goal, base_prompt, json_schema, query
from utils import searcher_cb, surfer_cb, clear_all
import sys
import os
from functools import partial
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llm_surfer.llm_surfer import LLMSurfer  # noqa: E402

st.title("LLM Surfer")

if st.session_state.get('CLIENT') is None:
    st.session_state['CLIENT'] = OpenAI(api_key=st.secrets['openai']["open_ai_key"])

research_goal = st.text_area("Research Goal", value=research_goal, height=500)
base_prompt = st.text_area("Base Prompt", value=base_prompt, height=500)
query = st.text_input("Query", value=query)
max_results = st.number_input("Max Results", min_value=1, max_value=2000, value=5)

if st.button("Surf 🏄‍♀️"):
    with st.spinner("Surfing..."):
        search_pbar = st.progress(0, text='Collecting relevant documents...')
        try:
            llm_surfer = LLMSurfer(
                client=st.session_state['CLIENT'],
                llm_name="gpt-4o-mini",
                research_goal=research_goal,
                base_prompt=base_prompt,
                json_schema=json_schema,
                query=query,
                args=[
                    "--headless",
                    "--mute-audio",
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    "--disable-blink-features=AutomationControlled",
                    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
                ],
                max_results=max_results,
                searcher_cb=partial(searcher_cb, pbar=search_pbar),
                surfer_cb=partial(surfer_cb)
            )
            df, output_path = llm_surfer()
            if st.session_state.get('RESULTS') is None:
                st.session_state['RESULTS'] = df
            st.write(st.session_state['RESULTS'])
            with open(output_path, "rb") as f:
                st.download_button("Download Results", data=f, file_name=output_path.split('/')[-1], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            time.sleep(60)
            clear_all()

if st.button("New search"):
    clear_all()

st.markdown("<footer><small>Assembed by Peter Nadel | Tufts University | Tufts Technology Services | Reserch Technology</small></footer>", unsafe_allow_html=True) 