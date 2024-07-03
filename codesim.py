# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("sachin.txt", "utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

query = "When Dhoni played international cricket last time?"
docs = db.similarity_search(query)

print(docs[0].page_content)

print(docs)