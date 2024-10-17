# utils/vectorstore.py => Pinecone에 데이터 추가하기
# 우리가 가진 자료를 집어넣는 것만 하는 파일임.
# pdf가 아니라면 PyMuPDFLoader가 아닌 다른 거 써야 함
# 실행할 때마다 데이터가 추가됨. 같은 거 누적되면 안된다.

import os
from dotenv import load_dotenv # OpenAIEmbeddings() 쓰려면 키가 있어야
from langchain_community.document_loaders.csv_loader import CSVLoader # 랭체인 CSV 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Pinecone index_name과 같아야 함
index_name = os.environ.get('INDEX_NAME') # .env 파일에서 가져올 때 쓰는 코드

# 1. Load CSV using file path string
loader = CSVLoader(file_path="./database.csv", encoding='utf-8')
docs = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 3. Embed Setting
embeddings = OpenAIEmbeddings()

# 4. Add Data to Index
PineconeVectorStore.from_documents(
    documents=split_docs, 
    embedding=embeddings,
    index_name=index_name
)