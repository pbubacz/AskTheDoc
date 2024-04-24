import streamlit as st
from local_lib import extract_text, num_tokens_from_string, get_ai_response, get_improved_prompts, check_user_prompt, get_prompt_from_template

def initialize_session_state():
    default_values = {
        'temperature': 0.0,
        'system_prompt': """You are a well-established scientific expert in the field of microbiology. 
Your language of communication is specialized and adapted to the needs of other scientists in this field. 
Please provide a detailed explanation of the question using only provided document as a reference.
Provide your response in Polish.""",
        'user_prompt_template': "# QUESTION:\n{question}\n# DOCUMENT:\n{document}",
        'main_prompt': "",
        'response': "",
        'question': "",
        'document': "",
        'model': "GPT-4-Turbo",
        'expander': True,
        'use_local_pdf': False
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar():
    with st.sidebar:
        st.session_state['model'] = st.selectbox(
            "Choose GPT model",
            ("GPT-4-Turbo", "GPT-3.5-Turbo", "GPT-4"),
            index=["GPT-4-Turbo", "GPT-3.5-Turbo", "GPT-4"].index(st.session_state['model'])
        )
        st.session_state['use_local_pdf'] = st.checkbox("Use local PDF converter", value=st.session_state['use_local_pdf'])
        st.session_state['system_prompt'] = st.text_area("System prompt", value=st.session_state['system_prompt'], height=350)
        st.session_state['user_prompt_template'] = st.text_area("User prompt template", value=st.session_state['user_prompt_template'], height=150)
        st.session_state['temperature'] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state['temperature'], step=0.1)

        
def file_upload():
    with st.expander("Upload file", expanded=st.session_state['expander']):
        uploaded_file = st.file_uploader("Upload file")        
        if st.button("Read file", key="read_file") and uploaded_file is not None:    
            with st.spinner("Converting...⏳"):
                try:
                    st.session_state['document'] = extract_text(uploaded_file, st.session_state['use_local_pdf'])                    
                except Exception as e:
                    st.error(f"Failed to process document: {e}")

def display_document():
    texts = st.session_state['document']
    if texts:
        with st.expander("Show document"):            
            st.text_area('Document content', texts, height=500)
            st.divider()
            st.write("Token number:", num_tokens_from_string(texts))

def ask_question():
    st.session_state['question'] = st.text_area("Ask your question", height=200)    
    if st.session_state['question']!="":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send to AI", key="ask_button"):
                if not check_user_prompt(st.session_state['user_prompt_template']):
                    st.error("User prompt template must contain {question} and {document} placeholders.")
                else:
                    with st.spinner("Sending to AI ⏳"):
                        prompt = get_prompt_from_template(st.session_state['user_prompt_template'],st.session_state['question'],st.session_state['document'])
                        st.session_state['response'] = get_ai_response(st.session_state['system_prompt'], prompt, st.session_state['temperature'], st.session_state['model'])

        with col2:
            if st.button("Propose better prompt", key="improve_button"):
                with st.spinner("Sending to AI ⏳"):            
                    st.session_state['response'] = get_improved_prompts(st.session_state['question'], st.session_state['temperature'], st.session_state['model'])

def display_response():
    if st.session_state['response']:
        st.markdown(st.session_state['response'], unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    initialize_session_state()
    sidebar()
    file_upload()
    display_document()
    ask_question()
    display_response()

if __name__ == "__main__":
    main()