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
        
    from pydantic import BaseModel, Field
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    # 테스트용 코드
    retrieval_grader = grade_prompt | structured_llm_grader

    # 241017 16:03
    ### Generate

    from langchain import hub # 좋은 프롬프트들을 가져와서 뜨게
    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt") # 이게 가져오는 프롬프트 이름임

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    ### Question Re-writer

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )


    question_rewriter = re_write_prompt | llm | StrOutputParser()
    ### Search

    from langchain_community.tools.tavily_search import TavilySearchResults

    web_search_tool = TavilySearchResults(max_results=3)

    # 24-10-21 11:12
    from typing import List

    from typing_extensions import TypedDict

    # 사전 세팅 단계
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
        """

        question: str
        generation: str
        web_search: str
        documents: List[str]
        
    from langchain.schema import Document

    def retrieve(state):
        """
        Retrieve documents from the CSV context.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE FROM CSV---")
        question = state["question"]

        # CSV에서 문서 검색
        docs = retriever.invoke(question)
        
        # 검색된 문서에 source 정보를 추가하여 반환
        documents = [Document(page_content=doc.page_content, metadata={"source": "CSV"}) for doc in docs]

        return {"documents": documents, "question": question, "source": "CSV"}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results and source as 'web'
        """
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        # Web search 실행
        docs = web_search_tool.invoke({"query": question})
        
        # 웹 검색 결과를 새로운 문서로 추가 (source 정보를 각 문서의 metadata에 추가)
        for d in docs:
            web_result = Document(page_content=d["content"], metadata={"source": "web"})
            documents.append(web_result)

        return {"documents": documents, "question": question, "source": "web"}




    ### Edges

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call ('generate_answer' or 'transform_query')
        """
        print("---ASSESS GRADED DOCUMENTS---")
        documents = state["documents"]

        # 문서가 있는지 여부에 따라 결정
        if documents:
            # 문서가 있으면 답변을 생성
            print("---DECISION: GENERATE ANSWER---")
            return "generate_answer"  # 문서가 있으면 바로 답변 생성
        else:
            # 문서가 없으면 질문을 변환하여 웹 검색
            print("---DECISION: TRANSFORM QUERY AND SEARCH---")
            return "transform_query"  # 문서가 없으면 질문 변환으로 이동


    def generate_answer(state):
        """
        Generate answer based on whether relevant movie exists in the context.
        """
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]

        # 문서가 없는 경우 먼저 처리
        if not documents:
            return {
                "documents": documents,
                "question": question,
                "generation": "죄송합니다. 관련 영화가 없습니다. 다른 영화를 추천 받으시겠어요?"
            }

        # 기본 답변 생성
        generation = rag_chain.invoke({"context": documents, "question": question})
        
        # 문서 소스 확인 및 메시지 추가
        has_csv = any(doc.metadata.get("source") == "CSV" for doc in documents)
        has_web = any(doc.metadata.get("source") == "web" for doc in documents)
        
        # 조건에 따라 적절한 메시지 추가
        if has_csv:
            generation += "\n\n해당 영화 및 사건과 관련된 정보를 POV Timeline에서 확인하실 수 있습니다."
        elif has_web:
            generation += "\n\nPOV Timeline은 해당 정보를 갖고 있지 않아, 웹에서 찾은 결과를 알려드렸습니다. 해당 영화가 궁금하시다면 웹 검색을 추천드립니다."

        return {
            "documents": documents,
            "question": question,
            "generation": generation
        }
        
    from langgraph.graph import END, StateGraph, START

    # 5. 워크플로우 설정
    workflow = StateGraph(GraphState)

    # 각 노드 정의
    workflow.add_node("retrieve", retrieve)  # CSV 검색 노드
    workflow.add_node("grade_documents", grade_documents)  # 문서 평가 노드
    workflow.add_node("generate_answer", generate_answer)  # 답변 생성 노드
    workflow.add_node("transform_query", transform_query)  # 질문 변환 노드
    workflow.add_node("web_search_node", web_search)  # 웹 검색 노드


    # 엣지 설정
    workflow.add_edge(START, "retrieve")  # 시작: CSV에서 검색
    workflow.add_edge("retrieve", "grade_documents")  # 문서 평가로 이동

    # 문서 평가 후 조건에 따라 질문 변환 또는 답변 생성
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,  # 문서가 적합한지 평가하여
        {
            "transform_query": "transform_query",  # 문서가 없으면 질문 변환
            "generate_answer": "generate_answer",  # 문서가 있으면 답변 생성
        },
    )

    # 질문 변환 후 웹 검색으로 이어지도록 설정
    workflow.add_edge("transform_query", "web_search_node")  # 변환된 질문으로 웹 검색
    workflow.add_edge("web_search_node", "generate_answer")  # 웹 검색 후 답변 생성

    # 답변 생성 후 종료
    workflow.add_edge("generate_answer", END)  # 답변 생성 후 종료

    # Compile
    app = workflow.compile()

    from pprint import pprint

    # Run
    inputs = {"question": user_input}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    return(value["generation"])