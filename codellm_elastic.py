import os
from typing import List
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from ast_finder import find_methods

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'myhugkey'

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.5, token='myhugkey'
    )

prompt_template = PromptTemplate(
    template="Given the following source code, generate a very simple usage example with maximun 5 lines of code and keywords, if source code is import no exmaples required :\n\nSource Code: {source_code}\n\nUsage Example:\n",
    input_variables=["source_code"]
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

dir = 'scan_code_base/retry/'
file_list = []
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=800, chunk_overlap=0
)

for root, dirs, files in os.walk(dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_list.append(file_path)
print(file_list)
docs :List[Document] = []

for file_path in file_list:
    if('.java' in file_path):
        try:
            class_methods = find_methods(file_path, 'java')
        except Exception as e:
            print('supressed')
        for class_info in class_methods:
            print(class_info['class_name'])
            for method_info in class_info['methods']:   
                print(class_info['class_name'] + '\n' + method_info['method_name'])             
                docs.append(Document(
                    page_content= class_info['class_name'] + '\n' + method_info['method_implementation'],
                    metadata={
                        'class_name': class_info['class_name']
                    }))

print('docs====',docs)
# for file_path in file_list:
#     loader = TextLoader(file_path, 'utf-8')
#     documents = loader.load()    
#     print(documents)
#     doc_in=python_splitter.split_documents(documents)
#     print(doc_in)
#     for doc in doc_in:
#         source_code = doc.page_content if hasattr(doc, 'page_content') else doc.get_page_content()
#         print(source_code)
#         # Generate example code
#         usage_example = chain.run({"source_code": source_code})
        
#          # Append the example code to the document's content
#         updated_content = source_code + f"\n\nUsage Example and keywords:\n{usage_example}"
#         print(updated_content)
#         # Update the document's content (assume `doc` has a method `set_content()` to update its content)
#         if hasattr(doc, 'page_content'):
#             doc.page_content = updated_content
#         else:
#             doc.set_page_content(updated_content)
#         print(doc)
#     docs_list.append(doc_in)





# docs = [doc for sub_list in docs_list for doc in sub_list]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.from_documents(docs, embeddings)

db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="https://0347233eb5574af1b83750c10fdb1982.us-central1.gcp.cloud.es.io:443",
    es_api_key="myelakey",
    index_name="rag-java" 
)

# db = ElasticsearchStore(
#     es_url="https://0347233eb5574af1b83750c10fdb1982.us-central1.gcp.cloud.es.io:443",
#     es_api_key="myelakey",
#     index_name="rag-java",
#     embedding=embeddings
# )

# -------------------------------------------------------------------------
# ------------------------------------------------------------------------
# RETRIVEAL STARTS HERE
# -------------------------------------------------------------------------
# ------------------------------------------------------------------------

query = "why we need polysantiago Retry? How is this different from spring provided retry logic?"


# prompt = hub.pull("rlm/rag-prompt")
system_prompt = (
    "You are an assistant for question-answering tasks for programming languages. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "provide the sample code on how to use the provided context."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

retriever = db.as_retriever(search_kwargs={"k": 4})
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

response = rag_chain_with_source.invoke(query)
print('------------------------------------------------')

# Format the output with clear labels
print(f"**Question:** {response['question']}")
print()
print(f"\n**Answer:** {response['answer']}")
print()
print(f"**Context Summary:**")
for page_content in response["context"]:
  print(f"\t- {page_content}")

print('------------------------------------------------')



# docs = db.similarity_search(query, )
# print(docs[0])

# promting(query=query, answer=docs[0].page_content)

# queryl = "In python, Write a function that takes two lists as input and returns a new list containing elements that are present in both lists."
# docs2 = db.similarity_search(queryl, )
# print(docs2[0])

# print(chain({"question":queryl}))

# print(db.similarity_search("How to check If given number is prim number?", )[0])

# print(db.similarity_search("How to arrange the number in a array in ascending order?", )[0])


