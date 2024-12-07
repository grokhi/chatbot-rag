import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.src.handlers.llm import LLMHandler
from backend.src.handlers.vector_db import VectorDBHandler
from backend.src.langgraph.nodes.grading import retrieval_grader

llm = LLMHandler().llm
retriever = VectorDBHandler().retriever


# ### Construct retriever ###
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
#     ),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(
#     documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# )
# retriever = vectorstore.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# rag_chain = rag_chain | retrieval_grader
### Statefully manage chat history ###
from collections import defaultdict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

store = defaultdict(ChatMessageHistory)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # if session_id not in store:
    #     store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# %%
web_search_tool = TavilySearchResults()
system = (
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    # "Dont forget to consider chat history when answering the question."
)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # MessagesPlaceholder("chat_history"),
        ("human", "Document: {document} Question: {question}"),
    ]
)


from backend.src.langgraph.nodes.query_transform import question_rewriter


def contextualize(state):
    q1 = "Какой город является столицей Туркмении?"
    q2 = "Что насчет Удмуртской республики?"
    q1 = "What is the weather in sf?"
    q2 = "What about new york?"

    #################### Round 1
    documents = history_aware_retriever.invoke(
        {"input": q1, "chat_history": store["abc123"].messages}
    )
    # rm duplicates
    documents = list({frozenset(d.metadata.items()): d for d in documents}.values())

    filtered_docs = []
    web_search = False

    for doc in documents:
        result = retrieval_grader.invoke({"question": q2, "document": doc.page_content})
        if result.binary_score == "yes":
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        web_search = True

    if web_search:
        web_search_tool = TavilySearchResults()
        context_question = question_rewriter.invoke(
            {"question": q1, "chat_history": store["abc123"].messages}
        )
        docs = web_search_tool.invoke(context_question)
        documents = [Document(d["content"]) for d in docs]

    # print(f1)
    # f11 = question_answer_chain.invoke(
    #     {
    #         "input": q1,
    #         "context": f1,
    #         "chat_history": [HumanMessage(q1)],
    #     }
    # )
    # print(f11)

    f111 = conversational_rag_chain.invoke(
        {"input": q1},
        config={"configurable": {"session_id": "abc123"}},  # constructs a key "abc123" in `store`.
    )
    print(f111)

    #################### Round 2
    print("------------------------------------")

    documents = history_aware_retriever.invoke(
        {"input": q2, "chat_history": store["abc123"].messages}
    )
    # rm duplicates
    documents = list({frozenset(d.metadata.items()): d for d in documents}.values())

    filtered_docs = []
    web_search = False

    for doc in documents:
        result = retrieval_grader.invoke({"question": q2, "document": doc.page_content})
        if result.binary_score == "yes":
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        web_search = True

    if web_search:
        web_search_tool = TavilySearchResults()
        context_question = question_rewriter.invoke(
            {"question": q2, "chat_history": store["abc123"].messages}
        )
        docs = web_search_tool.invoke(context_question)
        documents = [Document(d["content"]) for d in docs]

    """План
     - Теперь грейдер получает документы с памятью об истории чата.
     - Необходимо просто записать в стейт занчение web search и попросить начать поиск в интернете если релевантных доков не найдено
     - При этом поисковик тоже должен помнить историю чата
    """

    # n11 = question_answer_chain.invoke(
    #     {
    #         "input": q2,
    #         "context": n1,
    #         "chat_history": [HumanMessage(q1), HumanMessage(q2)],
    #     }
    # )

    # print(n11)

    n2 = conversational_rag_chain.invoke(
        {"input": q2},
        config={"configurable": {"session_id": "abc123"}},
    )

    print(n2)

    pass
