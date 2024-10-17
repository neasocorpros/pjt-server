# utils/llm.py

import os

from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_pinecone import PineconeVectorStore

text = '''
    넌 질문-답변을 도와주는 AI 영화 추천기야.
    아래 제공되는 Context를 통해서 사용자 Question에 대해 답을 해줘야해.

    Context에는 직접적으로 없어도, 추론하거나 계산할 수 있는 답변은 최대한 만들어 봐.
    만약 우리가 제공한 csv 파일에 관련 영화가 존재하지 않는다면, "죄송합니다. 관련 영화가 없습니다. 다른 영화를 추천 받으시겠어요?"라는 답변을 생성해줘.

    답은 적절히 \n를 통해 문단을 나눠줘 한국어로 만들어 줘. 
    # Question:
    {question}

    # Context:
    {context}


    # Answer:
    '''
def query_llm(user_input):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name = os.environ.get('INDEX_NAME'),
        embedding=OpenAIEmbeddings()
    )
    
    # 5. Retrieve
    retriever = vectorstore.as_retriever()

    # 6. Prompting
    prompt = PromptTemplate.from_template(text)

    # 7. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    # 8. Chain
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    ans = chain.invoke(user_input)
    return ans
