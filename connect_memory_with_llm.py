from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation
from typing import Optional, List
from pydantic import PrivateAttr


class MistralLLM(LLM):
    model_name: str
    token: str
    temperature: float = 0.7

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = InferenceClient(model=self.model_name, token=self.token)

    @property
    def _llm_type(self) -> str:
        return "custom-mistral"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.text_generation(
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=512
        )
        return response.strip()




import os 

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# Setup LLM (Mistral with HuggingFace)


huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.5,
        model_kwargs = {"token": HF_TOKEN,
                        "max_length": 512}
    )

    return llm

# Connect LLM with FAISS and create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template) : 
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
llm = MistralLLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

qa_chain= RetrievalQA.from_chain_type(
    llm= llm,
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents= True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query 

user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("Response: ", response['result'])
print("Source Documents: ", response['source_documents'])