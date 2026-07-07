import os
import math
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, END, StateGraph


load_dotenv()

app = FastAPI(title="Network Troubleshooting Agent")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
diagnosis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
DOCS_DIR = Path("docs")

class ChatRequest(BaseModel):
    question: str


class DiagnosisResult(BaseModel):
    problem_type: str = Field(description="네트워크 장애 유형")
    possible_causes: List[str] = Field(description="가능한 원인 후보 목록")
    recommended_commands: List[str] = Field(description="사용자가 확인할 수 있는 점검 명령어 목록")
    next_question: str = Field(description="추가 진단을 위해 사용자에게 물어볼 다음 질문")

parser = PydanticOutputParser(pydantic_object=DiagnosisResult)

PROBLEM_TYPES = [
    "SSH_CONNECTION_FAILED",
    "DHCP_LEASE_FAILED",
    "DNS_RESOLUTION_FAILED",
    "PING_CONNECTIVITY_CHECK",
    "INTERNET_CONNECTION_FAILED",
    "FIREWALL_OR_PORT_BLOCKED",
    "GATEWAY_OR_ROUTING_ISSUE",
    "UNKNOWN_NETWORK_ISSUE",
]


class SimpleVectorStore:
    def __init__(self):
        self.docs: List[dict] = []

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-10)

    def add_documents(self, docs: List[Document]) -> int:
        texts = [doc.page_content for doc in docs]
        embeddings = embedding_model.embed_documents(texts)

        for doc, embedding in zip(docs, embeddings):
            self.docs.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embedding,
                }
            )

        return len(self.docs)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if not self.docs:
            return []

        query_embedding = embedding_model.embed_query(query)

        scored_docs = [
            (self._cosine_sim(query_embedding, doc["embedding"]), doc)
            for doc in self.docs
        ]

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"],
            )
            for _, doc in scored_docs[:k]
        ]

    def clear(self):
        self.docs = []


rag_store = SimpleVectorStore()
rag_ready = False


def load_rag_documents():
    """
    docs 폴더의 Markdown 문서를 읽고 SimpleVectorStore에 저장한다.
    """
    global rag_ready

    if not DOCS_DIR.exists():
        rag_ready = False
        return 0

    documents = []

    for file_path in DOCS_DIR.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_path.name},
            )
        )

    if not documents:
        rag_ready = False
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    split_documents = text_splitter.split_documents(documents)

    rag_store.clear()
    count = rag_store.add_documents(split_documents)
    rag_ready = True

    return count


rag_document_count = load_rag_documents()


@tool
def network_diagnosis_tool(question: str) -> str:
    """
    사용자의 네트워크 장애 질문을 기반으로 문제 유형 분류
    """
    messages = [
        SystemMessage(
            content=(
                "당신은 네트워크 장애 유형을 분류하는 도구입니다. "
                "사용자의 문장 전체 의미를 보고 가장 적절한 장애 유형 하나만 선택하세요.\n\n"
                "중요 규칙:\n"
                "- 단순 키워드 포함 여부만 보지 마세요.\n"
                "- '~문제는 없다', '~는 된다', '~는 아님' 같은 부정 표현을 고려하세요.\n"
                "- 오타가 있어도 문맥상 의미를 추론하세요. 예: 게이드웨이 → 게이트웨이\n"
                "- 반드시 아래 목록 중 하나만 출력하세요.\n"
                "- 설명 문장은 쓰지 말고, 유형 이름만 출력하세요.\n"
                "- 네트워크 장애와 관련 없거나 분류가 어렵다면 UNKNOWN_NETWORK_ISSUE를 출력하세요.\n\n"
                f"선택 가능한 유형: {PROBLEM_TYPES}"
            )
        ),
        HumanMessage(content=question),
    ]
    response = diagnosis_llm.invoke(messages)

    result = (
        response.content
        .strip()
        .replace("`", "")
        .replace('"', "")
        .replace("'", "")
    )

    for problem_type in PROBLEM_TYPES:
        if problem_type in result:
            return problem_type
        
    return "UNKNOWN_NETWORK_ISSUE"


@tool
def command_recommendation_tool(problem_type: str) -> List[str]:
    """
    네트워크 장애 유형에 따라 사용자가 확인할 수 있는 점검 명령어를 추천한다.
    """
    command_map = {
        "SSH_CONNECTION_FAILED": [
            "ping <server_ip>",
            "systemctl status sshd",
            "ss -tulnp | grep :22",
            "firewall-cmd --list-all",
        ],
        "DHCP_LEASE_FAILED": [
            "ipconfig /all",
            "ip addr",
            "nmcli dev show",
            "journalctl -u NetworkManager --no-pager",
        ],
        "DNS_RESOLUTION_FAILED": [
            "nslookup google.com",
            "ping 8.8.8.8",
            "cat /etc/resolv.conf",
            "systemd-resolve --status",
        ],
        "PING_CONNECTIVITY_CHECK": [
            "ping <target_ip>",
            "tracert <target_ip>",
            "ipconfig /all",
            "route print",
        ],
        "INTERNET_CONNECTION_FAILED": [
            "ping 8.8.8.8",
            "nslookup google.com",
            "ipconfig /all",
            "tracert google.com",
        ],
        "FIREWALL_OR_PORT_BLOCKED": [
            "firewall-cmd --list-all",
            "ss -tulnp",
            "netstat -ano",
            "telnet <server_ip> <port>",
        ],
        "GATEWAY_OR_ROUTING_ISSUE": [
            "ip route",
            "route print",
            "ping <gateway_ip>",
            "traceroute 8.8.8.8",
        ],
        "UNKNOWN_NETWORK_ISSUE": [
            "ipconfig /all",
            "ip addr",
            "ping 8.8.8.8",
        ],
    }
    return command_map.get(problem_type, command_map["UNKNOWN_NETWORK_ISSUE"])


@tool
def rag_search_tool(question: str) -> str:
    """
    사용자의 질문과 관련된 네트워크 트러블슈팅 문서 검색
    """
    if not rag_ready:
        return "검색 가능한 RAG 문서가 없습니다."
    
    documents = rag_store.similarity_search(question, k=3)

    if not documents:
        return "관련 문서를 찾지 못했습니다."
    
    results = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        content = document.page_content.strip()

        results.append(
            f"[문서 {index}] source: {source}\n{content}"
        )
    return "\n\n".join(results)


class AgentState(TypedDict):
    question: str
    problem_type: str
    recommended_commands: List[str]
    rag_result: str
    answer: str
    structured_result: Dict[str, Any]
    diagnosis_tool_result: str
    command_tool_result: List[str]


def diagnose_node(state: AgentState) -> dict:
    """
    사용자 질문을 네트워크 장애 유형으로 분류하는 노드
    """
    diagnosis_type = network_diagnosis_tool.invoke(
        {"question": state["question"]}
    )
    return{
        "problem_type": diagnosis_type,
        "diagnosis_tool_result": diagnosis_type,
    }


def route_by_problem_type(state: AgentState) -> str:
    """
    장애 유형에 따라 다음 노드를 결정하는 조건부 분기 함수
    """
    if state["problem_type"] == "UNKNOWN_NETWORK_ISSUE":
        return "clarification"
    return "known_problem"


def rag_node(state: AgentState) -> dict:
    """
    사용자 질문과 관련된 RAG 문서를 검색하는 노드
    """
    rag_result = rag_search_tool.invoke(
        {"question": state["question"]}
    )
    return{
        "rag_result": rag_result,
    }


def command_node(state: AgentState) -> dict:
    """
    장애 유형에 따라 점검 명령어를 추천하는 노드
    """
    recommended_commands = command_recommendation_tool.invoke(
        {"problem_type": state["problem_type"]}
    )
    return{
        "recommended_commands": recommended_commands,
        "command_tool_result": recommended_commands,
    }


async def generate_answer_node(state: AgentState) -> dict:
    """
    진단 결과, RAG 검색 결과, 추천 명령어를 바탕으로 최종 답변을 생성하는 노드
    """
    messages = [
        SystemMessage(
            content=(
                "당신은 네트워크 트러블슈팅을 도와주는 AI Assistant입니다. "
                "사용자의 네트워크 장애 상황을 듣고 가능한 원인과 다음 확인 단계를 "
                "쉽고 간단하게 설명하세요. "
                "아직 Memory는 연결되지 않았으므로 "
                "LangGraph 기반 진단 흐름과 RAG 검색 결과를 함께 활용합니다.\n\n"
                f"Network Diagnosis Tool이 분류한 장애 유형은 다음과 같습니다: {state['problem_type']}\n"
                "반드시 problem_type에는 위 장애 유형을 그대로 사용하세요.\n\n"
                f"Command Recommendation Tool이 추천한 명령어는 다음과 같습니다: {state['recommended_commands']}\n"
                "반드시 recommended_commands에는 위 명령어 목록을 그대로 사용하세요.\n\n"
                f"RAG Search Tool이 검색한 문서 내용은 다음과 같습니다:\n{state['rag_result']}\n\n"
                "가능한 원인과 다음 확인 단계는 위 RAG 검색 결과를 참고해서 작성하세요.\n\n"
                "응답은 반드시 아래 형식 지침을 따르세요.\n"
                f"{parser.get_format_instructions()}\n\n"
                "주의사항:\n"
                "- 반드시 JSON 형식으로만 답변하세요.\n"
                "- 마크다운 코드블록은 사용하지 마세요.\n"
                "- recommended_commands에는 실제 점검에 사용할 수 있는 명령어를 넣으세요.\n"
                "- next_question에는 추가 진단을 위해 사용자에게 물어볼 질문을 넣으세요.\n"
            )
        ),
        HumanMessage(content=state["question"]),
    ]

    response = await llm.ainvoke(messages)

    parsed_result = parser.parse(response.content)
    parsed_result.problem_type = state["problem_type"]
    parsed_result.recommended_commands = state["recommended_commands"]

    answer = (
        f"진단 유형은 {parsed_result.problem_type}입니다. "
        f"가능한 원인은 {', '.join(parsed_result.possible_causes)}입니다. "
        f"먼저 {', '.join(parsed_result.recommended_commands)} 명령어를 확인해보세요. "
        f"추가로 확인할 점은 다음과 같습니다: {parsed_result.next_question}"
    )
    return {
        "answer": answer,
        "structured_result": parsed_result.model_dump(),
    }


def clarification_node(state: AgentState) -> dict:
    """
    장애 유형을 판단하기 어려운 경우 추가 정보를 요청하는 노드
    """
    recommended_commands = command_recommendation_tool.invoke(
        {"problem_type": "UNKNOWN_NETWORK_ISSUE"}
    )

    structured_result = DiagnosisResult(
        problem_type="UNKNOWN_NETWORK_ISSUE",
        possible_causes=[
            "질문 정보가 부족하여 원인을 특정할 수 없습니다."
        ],
        recommended_commands = recommended_commands,
        next_question = "구체적인 증상, 오류 메시지, 사용 중인 OS, 네트워크 연결 상황을 알려주세요."
    )

    answer = (
        "현재 질문만으로는 정확한 네트워크 장애 유형을 판단하기 어렵습니다. "
        "어떤 상황에서 문제가 발생하는지 조금 더 구체적으로 알려주세요. "
        f"추가로 확인할 점은 다음과 같습니다: {structured_result.next_question}"
    )

    return{
        "problem_type": "UNKNOWN_NETWORK_ISSUE",
        "recommended_commands": recommended_commands,
        "command_tool_result": recommended_commands,
        "rag_result": "",
        "answer": answer,
        "structured_result": structured_result.model_dump(),
    }

workflow = StateGraph(AgentState)

workflow.add_node("diagnose_node", diagnose_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("command_node", command_node)
workflow.add_node("generate_answer_node", generate_answer_node)
workflow.add_node("clarification_node", clarification_node)

workflow.add_edge(START, "diagnose_node")

workflow.add_conditional_edges(
    "diagnose_node",
    route_by_problem_type,
    {
        "known_problem": "rag_node",
        "clarification": "clarification_node"
    },
)

workflow.add_edge("rag_node", "command_node")
workflow.add_edge("command_node", "generate_answer_node")
workflow.add_edge("generate_answer_node", END)
workflow.add_edge("clarification_node", END)

agent_graph = workflow.compile()


@app.get("/api/health")
async def health():
    return {
        "success": True,
        "message":"Network Troubleshooting Agent server is running",
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        final_state = await agent_graph.ainvoke(
            {
                "question": req.question,
            }
        )
        return{
            "success": True,
            "question": req.question,
            "answer": final_state.get("answer", ""),
            "structured_result": final_state.get("structured_result", {}),
            "diagnosis_tool_result": final_state.get(
                "diagnosis_tool_result",
                final_state.get("problem_type", "UNKNOWN_NETWORK_ISSUE"),
            ),
            "command_tool_result": final_state.get(
                "command_tool_result",
                final_state.get("recommended_commands", []),
            ),
            "rag_result": final_state.get("rag_result", ""),
            "rag_document_count": rag_document_count,
            "graph_used": True,
            "graph_flow": [
                "START",
                "diagnose_node",
                "conditional_edge",
                "rag_node or clarification_node",
                "command_node",
                "generate_answer_node",
                "END",
            ],
            "model": "gpt-4o-mini",
        }
        
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

