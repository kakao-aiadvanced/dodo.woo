from typing import List

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from typing_extensions import TypedDict

# export
tavily = TavilyClient(api_key="")

st.set_page_config(
    page_title="RAG Assistant Of Kakao AI Advanced",
    page_icon=":yellow_heart:",
)

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


def main():
    ### Index
    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()

    ### Relvance Checker
    system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    ### Generate
    system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question with reference. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise. Return the answer and reference as a JSON format with a double key 'answer' and 'reference'"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    # Chain
    rag_chain = prompt | llm | JsonOutputParser()

    ### Hallucination Checker
    system = """You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

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
        hallucination: str
        documents: List[str]


    ### Nodes

    def retrieve(state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        print(question)
        print(documents)
        return {"documents": documents, "question": question}


    def generate(state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        hallucination = "NO"

        if "generation" in state:
            hallucination = "YES"

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation, "hallucination": hallucination}


    def web_search(state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = None
        if "documents" in state:
            documents = state["documents"]

        # Web search
        docs = tavily.search(query=question)['results']
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        print(documents)
        return {"documents": documents, "question": question, "web_search": "YES"}


    def check_relevance(state):
        """
        Determines whether the retrieved documents are relevant to the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []

        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade.lower() == "yes":
                print("--RELEVANT--")
                filtered_docs.append(d)
            else:
                print("--NOT RELEVANT--")
                continue


        return {"documents": filtered_docs, "question": question}



    ### Edges

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---DECIDE TO GENERATE---")
        documents = state["documents"]

        if not documents:
            if "web_search" in state:
                print("---failed: not relevant---")
                return "end"
            else:
                print("---WEB SEARCH---")
                return "websearch"

        print("---GENERATE---")
        return "generate"


    ### Conditional edge


    def check_hallucination(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]
        hallucination = state["hallucination"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            return "useful"
        elif hallucination == "YES":
            print("failed: hallucination")
            return "not supported"
        else:
            return "retry"


    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("generate", generate)


    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "check_relevance")
    workflow.add_conditional_edges("check_relevance", decide_to_generate, {
        "websearch": "websearch",
        "generate": "generate",
        "end": END,
    })

    workflow.add_edge("websearch", "check_relevance")

    workflow.add_conditional_edges(
        "generate",
        check_hallucination,
        {
            "not supported": END,
            "useful": END,
            "retry": "generate",
        },
    )

    # Compile
    app = workflow.compile()


    # ----------------------------------------------------------------------
    # Streamlit 앱 UI
    st.title("RAG assistant")
    st.caption("🚀 KAKAO AI Advanced")
    with st.sidebar:
        "[강의 바로가기](https://docs.google.com/spreadsheets/d/14fRbnVIsxdiS3uDnNua0IqkkpT0BBUrjKIwl5iVryuc)"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "궁금한게 뭔가요?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Generating Report"):
            inputs = {"question": prompt}
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
            final_report = value["generation"]

            st.session_state.messages.append({"role": "assistant", "content": final_report})
            st.chat_message("assistant").write(final_report)


main()
