from langchain_community.document_loaders import WebBaseLoader

url = "https://sklearn.org/stable/auto_examples/classification/plot_lda_qda.html"

loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0].page_content)