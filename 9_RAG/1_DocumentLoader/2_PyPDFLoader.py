from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader("OS_UNIT_1.pdf")
docs = loader.load()

prompt = prompt = PromptTemplate(
    template="Generate 3 major important questions along with answers form following document: \n\n {document}",
    input_variables=['document']
)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)   
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser 
chain_result = chain.invoke({'document': docs[0].page_content})
print(chain_result)  # Print the first 500 characters of the first page
