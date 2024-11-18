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
    당신은 질문-답변을 도와주는 AI 영화 추천기입니다.
    아래 제공되는 Context를 통해서 사용자 Question에 대해 답변하십시오.

    아래 해당 내용 규칙에 맞게 답변을 작성해주십시오.

    1) 기본 지침:
    - 당신은 전문 영화 분석가이자 큐레이터로서, 사용자 질문과 가장 관련성이 높은 영화 콘텐츠를 추천해야 합니다.
    - 대한민국 현대사와 직간접적으로 관련된 영화들을 추천해야 합니다. 해외에서 일어난 사건을 다룬다 하더라도 한국의 현대사와 직간접적으로 관련된 영화만을 추천해야 합니다.
    - 사용자의 질문에서 1) 실제 역사 관련 정보(특정 역사적 사건이나 시대, 인물, 단체, 공간 등) 혹은 2) 영화 관련 정보(장르, 분위기, 출연 배우, 감독 등) 추출해내고, 이를 바탕으로 가장 적합한 영화 혹은 영화들을 추천해야 합니다.
    - 가급적 실제 사건을 배경으로 하는 영화들을 추천해야 합니다. 다만 특정 영화가 다루는 주제/소재가 한국 현대사와 시대 배경을 매우 잘 묘사하고 있다거나, 실제 역사적 사건에 상상력을 가미했거나, 비유적 기법으로 해당 사건을 유추하게 만들어졌을 경우에는 해당 영화를 추천해도 무방합니다.
    - 사용자 질문이 한 연도가 아닌 시대와 관련될 경우, 0년부터 9년까지의 사건을 기준으로 답해야 합니다. 예를 들어 사용자가 "70년대 영화를 추천"해달라는 질문을 던질 경우, 1970년부터 1979년에 일어난 사건과 관련한 영화만 추천해야 합니다.
    - 제시된 제약 조건과 사용자 질문에 기반하여 최고의 답변을 생성하세요.

    2) 답변 제약조건:
    - 사용자 질문에 따른 추천 영화가 무엇인지 한두 문장으로 가장 먼저 밝히세요. 예시: 사용자 질문이 "80년대 영화 추천"일 경우, "80년대 한국영화 중에서 추천할 만한 작품으로는 **화려한 휴가**와 **택시운전사**가 있습니다." 등과 같이 밝힐 것.
    - 영화 제목이나 중요한 역사적 사건명 등 강조가 필요한 부분은 볼드체로 표기해 주십시오. 마크다운 형식입니다.
    - 한 문장의 길이는 되도록 한글로 40자(공백 포함) 이내로 작성해주십시오.
    - 추천하는 각 영화에 대해 400자 이내로 설명하세요. 해당 영화가 왜 사용자의 질문과 관련되는지 명확한 이유를 포함해야 합니다.
    - 실제 사건을 언급할 경우 해당 사건이 벌어진 년도를 명확히 언급해주십시오. 예를 들면, 5.18민주화운동(1980)이라고 명시해주십시오.
    - 영화의 역사적 의미를 설명함에 있어서, 해당 영화가 표면적으로 다루는 주제가 개인의 일상, 범죄 등 겉보기에 정치적 요인과 무관해 보이더라도, 해당 영화에 대한 전반적인 비평이 정치적 요소를 포함하고 있거나, 정치적으로 해석할 수 있는 여지가 강하거나, 거시적인 정치 요소에 영향을 받은 사건일 경우 해당 부분을 짚어주시기 바랍니다. 예를 들어, 영화 살인의추억의 경우 군사정권의 무능함과 폭력적 통치를 지적할 수 있습니다. 영화 '벌새'의 경우 한국사회가 그동안 자본주의적 경제성장에만 매몰되어, 성수대교 붕괴와 같은 사회적 참사와 개인의 상처를 만들어낸 측면을 지적할 수 있을 것입니다.
    - 그런 다음, 해당 영화가 갖는 함의(영화사적으로 차지하는 위치 혹은 실제 대한민국 역사에 있어서 갖는 의미)를 한두 줄로 요약해서 제시하세요.

    3) Vectorstore 설명:
    - 'eventYear'는 실제 역사적 사건이 발생한 연도를 가리킵니다.
    'event'는 영화에서 다루는 실제 대한민국 현대사 사건을 가리킵니다.
    'historyDescription'은 해당 사건 혹은 시대를 표현하는 구체적인 키워드들을 담고 있습니다.
    'title'은 해당 영화의 제목입니다.
    'genre'는 해당 영화의 장르 혹은 분위기를 의미합니다.
    'movieDescription'은 영화의 감독이나 주조연 배우 출연진을 나타냅니다.
    'synopsys'는 영화의 간략한 줄거리를 담고 있습니다.

    Context에는 직접적으로 없어도, 추론하거나 계산할 수 있는 답변은 최대한 만들어 보십시오.

    답은 적절히 \n를 통해 문단을 나누고, 한국어로 만들어 주십시오.

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
        Retrieve documents from the JSONL context.
        """
        print("---RETRIEVE FROM JSONL---")
        question = state["question"]

        # JSONL에서 문서 검색
        docs = retriever.invoke(question)
        
        # 검색된 문서들의 메타데이터 확인
        for doc in docs:
            print(f"Document metadata: {doc.metadata}")  # 디버깅용 로그 추가
        
        # 검색된 문서에 source 정보와 id 정보를 추가하여 반환
        documents = []
        for doc in docs:
            # 문서 내용에서 ID 추출 시도
            try:
                content = doc.page_content
                if '"id": ' in content:
                    movie_id = content.split('"id": ')[1].split(',')[0].strip()
                    doc.metadata['id'] = movie_id
            except Exception as e:
                print(f"Error extracting ID from content: {e}")
            
            doc.metadata['source'] = "JSONL"
            documents.append(doc)

        return {"documents": documents, "question": question, "source": "JSONL"}

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

        # 메타데이터에서 영화 ID 추출
        movie_ids = []
        for doc in documents:
            if doc.metadata and 'id' in doc.metadata:
                # float에서 int로 변환 후 문자열화
                movie_id = int(float(doc.metadata['id']))  # float -> int로 변환
                movie_ids.append(movie_id)
            
        print(f"Extracted movie IDs from metadata: {movie_ids}")  # 디버깅용 로그

        # 기본 답변 생성
        generation = rag_chain.invoke({"context": documents, "question": question})

        # 문서 소스 확인 및 메시지 추가
        has_jsonl = any(doc.metadata.get("source") == "JSONL" for doc in documents)
        has_web = any(doc.metadata.get("source") == "web" for doc in documents)
        
        # 조건에 따라 적절한 메시지 추가
        if has_jsonl:
            generation += "\n\n해당 영화 및 사건과 관련된 정보를 POV Timeline에서 확인하실 수 있습니다."
        elif has_web:
            generation += "\n\nPOV Timeline은 해당 정보를 갖고 있지 않아, 웹에서 찾은 결과를 알려드렸습니다. 해당 영화가 궁금하시다면 웹 검색을 추천드립니다."

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "movie_ids": movie_ids  # 추출된 영화 ID 목록 추가
        }
        
    from langgraph.graph import END, StateGraph, START

    # 5. 워크플로우 설정
    workflow = StateGraph(GraphState)

    # 각 노드 정의
    workflow.add_node("retrieve", retrieve)  # jsonl 검색 노드
    workflow.add_node("grade_documents", grade_documents)  # 문서 평가 노드
    workflow.add_node("generate_answer", generate_answer)  # 답변 생성 노드
    workflow.add_node("transform_query", transform_query)  # 질문 변환 노드
    workflow.add_node("web_search_node", web_search)  # 웹 검색 노드


    # 엣지 설정
    workflow.add_edge(START, "retrieve")  # 시작: jsonl에서 검색
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
    ans = app.invoke(inputs)

    return {
        'generation': ans,
    }

    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         # Node
    #         pprint(f"Node '{key}':")
    #         # Optional: print full state at each node
    #         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    #     pprint("\n---\n")

    # # Final generation
    # # 마지막 부분만 수정
    # if isinstance(value, dict) and movie_ids in value:
    #     # generate_answer 노드에서 반환된 경우
    #     return {
    #         "generation": value["generation"],
    #         "movie_ids": value["movie_ids"]
    #     }
    # else:
    #     # 다른 노드에서 반환된 경우
    #     return {
    #         "generation": value["generation"] if isinstance(value, dict) else value,
    #         "movie_ids": []
    #     }