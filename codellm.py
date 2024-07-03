import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    Language,
        RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'myhugkey'
def promting(query, answer):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    question = query

    # template = f"""Question: {question}
    # Answer: Let's think and make the answer in detailed."""

    template = f"""
    Question: {question}
    Here's some Python code:

    ```python
    {answer}
    ```

    What is the time and space complexity of this code?
    What are some potential optimizations that could be made?
    Can you provide a test case for this code?
    """

    print(template)

    prompt = PromptTemplate.from_template(template)

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.5, token='myhugkey'
    )
    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", output_key='answer', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm, memory=memory, answer=answer, question=question)

    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(chain(question=question, answer=answer))

dir = 'scan_code_base'
file_list = []
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=200, chunk_overlap=0
)

for root, dirs, files in os.walk(dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_list.append(file_path)
print(file_list)
docs_list = []
for file_path in file_list:
    loader = TextLoader(file_path, 'utf-8')
    documents = loader.load()
    docs_list.append(python_splitter.split_documents(documents))

docs = [doc for sub_list in docs_list for doc in sub_list]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

query = "python code to search If given number exists, otherwise return -1?"

llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.5, token='myhugkey'
    )
memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", output_key='answer', return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, memory=memory, retriever= db.as_retriever())

print(chain({"question":query}))

# docs = db.similarity_search(query, )
# print(docs[0])

# promting(query=query, answer=docs[0].page_content)

queryl = "In python, Write a function that takes two lists as input and returns a new list containing elements that are present in both lists."
docs2 = db.similarity_search(queryl, )
print(docs2[0])

print(chain({"question":queryl}))

print(db.similarity_search("How to check If given number is prim number?", )[0])

print(db.similarity_search("How to arrange the number in a array in ascending order?", )[0])


