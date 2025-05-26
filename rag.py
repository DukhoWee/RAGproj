import getpass
import streamlit as st
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for HuggingFace: ")

from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

st.title("RAG Chatbot")
st.markdown(
    """
    <style>
    /* 텍스트 입력 위젯 전체 컨테이너를 잡아서 */
    div[data-testid="stTextInput"] > div > div > input {
        /* 텍스트 색상 */
        color: #f0f0f0 !important;
        /* 배경색 */
        background-color: #000000 !important;
        /* 테두리 색상 */
        border: 2px solid #f0f0f0 !important;
    }

    /* 포커스 되었을 때 (예: 클릭했을 때) */
    div[data-testid="stTextInput"] > div > div > input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.session_state.messages = []
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

loader = TextLoader("profile.txt", encoding="utf-8")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=chunks)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
    
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

#import ipdb; ipdb.set_trace()
# Compile application and test
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()
st.session_state.graph = graph

from IPython.display import Image, display

png_data = graph.get_graph().draw_mermaid_png()
with open("graph3.png", "wb") as f:
    f.write(png_data)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("무엇이든 물어보세요", "")
    submit = st.form_submit_button("제출")
if submit and user_input:
    # 1) 사용자 메시지 쌓기
    st.session_state.messages.append({"role":"user","content":user_input})
    # 2) 빈 컨테이너 마련 (스트리밍 출력용)
    chat_placeholder = st.empty()
    # 3) graph.stream 호출
    for step in st.session_state.graph.stream(
        {"messages": st.session_state.messages},
        stream_mode="values",
    ):
        # 각 단계의 마지막 메시지를 렌더링
        msg = step["messages"][-1]
        role = msg.role if hasattr(msg, "role") else msg.type
        content = msg.content

        # 화면에 누적 출력
        chat_placeholder.markdown(f"**{role}**: {content}")
        # 세션 상태에도 저장해 두면 재실행 시에도 기록 유지
        st.session_state.messages.append({
            "role": role,
            "content": content
        })