from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader

loader = DirectoryLoader(
    path=r"C:\Users\Acer\OneDrive\Desktop\LangChainModel\9_RAG\1_DocumentLoader",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(f"Loaded {len(docs)} documents from directory.")