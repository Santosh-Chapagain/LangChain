from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    r"C:\Users\Acer\OneDrive\Desktop\LangChainModel\9_RAG\1_DocumentLoader\OS_UNIT_1.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10,

)
result = splitter.split_documents(documents)    
print(result[10])