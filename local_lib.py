import os
import re
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat, AnalyzeResult
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import tiktoken

load_dotenv()

MAX_CHARACTERS = 1500
NEW_AFTER_N_CHARS = 2000
COMBINE_UNDER_N_CHARS = 100

encoding = tiktoken.get_encoding("cl100k_base")

client_call = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

model_mapping = {
    "GPT-3.5-Turbo": os.getenv("AZURE_OPENAI_MODEL_GPT35"),
    "GPT-4": os.getenv("AZURE_OPENAI_MODEL_GPT4"),
    "GPT-4-Turbo": os.getenv("AZURE_OPENAI_MODEL_GPT4T"),
}

def get_env_var(var_name):
    return os.getenv(var_name)

def handle_error(e, message):
    return f"{message} \n{e}"

def handle_plain_or_octet_stream(file) -> str:
    return file.getvalue().decode('utf-8')

def handle_other(file) -> str:
    try:
        bytes_io = BytesIO(file.getvalue())
        elements = partition(file=bytes_io)        
        chunks = chunk_by_title(elements, multipage_sections=True, max_characters=MAX_CHARACTERS, new_after_n_chars=NEW_AFTER_N_CHARS, combine_text_under_n_chars=COMBINE_UNDER_N_CHARS)  
        out_text = ''.join(chunk.metadata.text_as_html if chunk.category == 'Table' else chunk.text for chunk in chunks)
        return clean_text(out_text)
    except Exception as e:
        return handle_error(e, "Error processing document:")

def handle_pdf_locally(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        texts = [page.extract_text() for page in pdf_reader.pages]
        return clean_text('\n'.join(texts))
    except Exception as e:
        return handle_error(e, "Error processing document:")

def handle_pdf_remotly(uploaded_file):   
    try:
        doc_intelligence_endpoint = get_env_var("DOCUMENTINTELLIGENCE_ENDPOINT")
        doc_intelligence_key = get_env_var("DOCUMENTINTELLIGENCE_API_KEY")
       
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=doc_intelligence_endpoint, credential=AzureKeyCredential(doc_intelligence_key)
        )    
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", 
            analyze_request=uploaded_file,
            content_type="application/octet-stream", 
            output_content_format=ContentFormat.MARKDOWN)       
        result: AnalyzeResult = poller.result()
        return clean_text(result.content)
    except Exception as e:
        return handle_error(e, "Error processing PDF document remotely:")

def extract_text(uploaded_file, use_local_pdf=False) -> str:
    try:
        file_type = uploaded_file.type
        if file_type in ['text/plain', 'application/octet-stream']:
            return handle_plain_or_octet_stream(uploaded_file)
        if file_type == 'application/pdf':
            if use_local_pdf:                
                return handle_pdf_locally(uploaded_file)
            else:
                return handle_pdf_remotly(uploaded_file)
        return handle_other(uploaded_file)
    except Exception as e:
        return handle_error(e, "Error extracting text:")

def num_tokens_from_string(string: str) -> int:
    return len(encoding.encode(string))

def get_ai_response(system_prompt, query, temperature, model): 
    try: 
        model = model_mapping.get(model)

        response = client_call.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}   
            ]   
        )
        return response.choices[0].message.content
    except Exception as e:
        return handle_error(e, "Error getting AI response. Try again.")

def get_improved_prompts(query, temperature, model):    
    system_prompt = "You are AI Specialist"
    prompt = (f"Could you help me improve the following prompt to ensure it elicits higher-quality responses from the GPT model? "
              f"I aim for a prompt that is succinct, clear, and specifically tailored to produce the most accurate answer. "
              f"Please incorporate all relevant guidelines from OpenAI's prompt engineering strategies. "
              f"Provide improved prompt in the original language.\n{query}")
    return get_ai_response(system_prompt, prompt, temperature, model)

def clean_text(text):
    text = re.sub('(?<=<table>)(.*?)(?=</table>)', lambda m: m.group(0).replace('\n', ' '), text, flags=re.DOTALL)
    patterns = {
        '\n+': '\n',
        ' +': ' ',
        r'\s<': '<',
        r'>\s': '>',
        r'\s\.': '.',
        r'\s,': ',',
        r'\s!': '!',
        r'\s\?': '?',
        r'\s:': ':',
        r'\s;': ';',
        r'\s\)': ')',
        r'\(\s': '(',
        r'\[\s': '[',
        r'\s\]': ']',
        r'\s\}': '}',
        r'\}\s': '}',
    }
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    text = text.replace('<table>', '\n<table>')
    return text