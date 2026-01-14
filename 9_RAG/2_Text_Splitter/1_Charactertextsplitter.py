from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(
    r"C:\Users\Acer\OneDrive\Desktop\LangChainModel\9_RAG\1_DocumentLoader\OS_UNIT_1.pdf")

documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator="",
)
result= splitter.split_documents(documents)
print(result[0])
