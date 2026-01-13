from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Summarize the following poem: \n\n {poem}",    
    input_variables=['poem']
)
parser = StrOutputParser()

chain = prompt | model | parser
# Load the local text file and return a list of Documents
loader = TextLoader("poem.txt", encoding="utf-8")
docs = loader.load()

result = chain.invoke({'poem': docs[0].page_content})

print(result)