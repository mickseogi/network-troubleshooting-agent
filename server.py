import os
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
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
PUBLIC_DIR = Path("public")

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="사용자 네트워크 장애 질문",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=40,
        description="대화 세션 ID",
    )


class DiagnosisResult(BaseModel):
    problem_type: str = Field(description="네트워크 장애 유형")
    possible_causes: List[str] = Field(description="가능한 원인 후보 목록")
    recommended_commands: List[str] = Field(description="사용자가 확인할 수 있는 점검 명령어 목록")
    next_question: str = Field(description="추가 진단을 위해 사용자에게 물어볼 다음 질문")
    user_facing_answer: str = Field(description="사용자에게 보여줄 자연스러운 최종 답변")

parser = PydanticOutputParser(pydantic_object=DiagnosisResult)

memory_stores: dict[str, InMemoryChatMessageHistory] = {}

middleware_sessions: dict[str, dict[str, Any]] = {}


def get_middleware_session(session_id: str) -> dict[str, Any]:
    """
    session_id별 middleware 상태를 가져옴
    없으면 새로 생성
    """
    if session_id not in middleware_sessions:
        middleware_sessions[session_id] = {
            "id": session_id,
            "logs": [],
            "model_call_count": 0,
            "tool_call_count": 0,
            "config": {
                "model_call_limit": 100,
                "tool_call_limit": 200,
                "log_retention": 100,
            },
        }
    
    return middleware_sessions[session_id]


def add_middleware_log(
    session_id: str,
    middleware: str,
    level: str,
    message: str,
    detail: Optional[str] = None,
) -> dict:
    """
    middleware 실행 로그를 session 별로 저장
    """
    session = get_middleware_session(session_id)

    log = {
        "id": f"{time.time()}-{uuid.uuid4().hex[:6]}",
        "middleware": middleware,
        "level": level,
        "message": message,
        "detail": detail,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    session["logs"].append(log)

    retention = session["config"].get("log_retention", 50)
    session["logs"] = session["logs"][-retention:]

    return log


def get_recent_middleware_logs(session_id: str, limit: int = 20) -> List[dict]:
    """
    최근 middleware 로그 반환
    """
    session = get_middleware_session(session_id)
    return session["logs"][-limit:]


def request_logging_middleware(session_id: str, question: str) -> None:
    """
    Agent 요청 시작 로그를 기록
    """
    add_middleware_log(
        session_id=session_id,
        middleware="requestLogging",
        level="info",
        message="Agent 요청 시작",
        detail=f"question={question}",
    )


def model_call_limit_middleware(session_id: str, purpose: str) -> None:
    """
    LLM 호출 횟수를 기록하고 계산
    """
    session = get_middleware_session(session_id)
    session["model_call_count"] += 1

    count = session["model_call_count"]
    limit = session["config"]["model_call_limit"]

    if count > limit:
        add_middleware_log(
            session_id=session_id,
            middleware="modelCallLimit",
            level="block",
            message=f"LLM 호출 한도 초과 ({count}/{limit})",
            detail=f"purpose={purpose}",
        )
        raise RuntimeError(f"LLM 호출 한도({limit}회)를 초과했습니다.")
    
    add_middleware_log(
        session_id=session_id,
        middleware="modelCallLimit",
        level="pass",
        message=f"LLM 호출 허용 ({count}/{limit})",
        detail=f"purpose={purpose}",
    )


def tool_call_limit_middleware(session_id: str, tool_name: str, tool_args: dict) -> None:
    """
    Tool 호출 횟수를 기록하고 제한
    """
    session = get_middleware_session(session_id)
    session["tool_call_count"] += 1

    count = session["tool_call_count"]
    limit = session["config"]["tool_call_limit"]

    if count > limit:
        add_middleware_log(
            session_id=session_id,
            middleware="toolCallLimit",
            level="block",
            message=f"Tool 호출 한도 초과 ({count}/{limit})",
            detail=f"tool={tool_name}, args={tool_args}",
        )
        raise RuntimeError(f"Tool 호출 한도({limit}회)를 초과했습니다.")
    
    add_middleware_log(
        session_id=session_id,
        middleware="toolCallLimit",
        level="pass",
        message=f"Tool 호출 허용 ({count}/{limit})",
        detail=f"tool={tool_name}, args={tool_args}",
    )


def agent_finish_logging_middleware(session_id: str, final_state: dict) -> None:
    """
    Agent 실행 완료 로그를 기록
    """
    add_middleware_log(
        session_id=session_id,
        middleware="agentFinishLogging",
        level="info",
        message="Agent 실행 완료",
        detail=(
            f"diagnosis={final_state.get('diagnosis_tool_result')}, "
            f"memory_saved={final_state.get('memory_saved')}, "
            f"graph_flow={final_state.get('graph_flow')}"
        ),
    )


def agent_error_logging_middleware(session_id: str, error: Exception) -> None:
    """
    Agent 실행 중 에러 로그를 기록
    """
    add_middleware_log(
        session_id=session_id,
        middleware="agentErrorLogging",
        level="error",
        message="Agent 실행 중 오류 발생",
        detail=str(error),
    )


def get_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    session_id별 대화 이력을 가져옴
    없으면 새로 생성
    """
    if session_id not in memory_stores:
        memory_stores[session_id] = InMemoryChatMessageHistory()
    return memory_stores[session_id]


def messages_to_json(messages: List[BaseMessage]) -> List[dict]:
    """
    Langchain Message 객체를 API 응답용 JSON 형태로 반환
    """
    return [
        {
            "role": message.type,
            "content": message.content,
        }
        for message in messages
    ]


def build_memory_context(messages: List[BaseMessage], max_messages: int = 4, max_content_length: int = 180) -> str:
    """
    최근 대화 이력을 LLM 프롬프트에 넣기 좋은 짧은 문자열로 변환
    """
    recent_messages = messages[-max_messages:]

    if not recent_messages:
        return "이전 대화 이력이 없습니다."
    lines = []

    for message in recent_messages:
        role = "사용자" if message.type == "human" else "AI"
        content = message.content

        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


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

INPUT_GUARD_TYPES = [     #bug_fix(#19): 일단은 일관되지 않은 버그 발생이니, 기존 코드에서 진단 전에 의미 없는 입력이 known_problem으로 가는걸 막는 방향으로 구현할 예정 
    "VALID_INPUT",
    "GENERAL_CHAT",
    "INVALID_INPUT",
]


def is_repeated_mieum_input(question: str) -> bool:
    """
    비정상적인 입력을 감지
    """
    compact_text = "".join(question.strip().split())

    if len(compact_text) < 30:
        return False

    mieum_count = compact_text.count("ㅁ")

    return mieum_count / len(compact_text) >= 0.8


def classify_current_input(question: str) -> str:
    """
    현재 질문만 보고 Agent 진단을 진행해도 되는 입력인지 판단
    Memory context는 사용안함
    """
    if is_repeated_mieum_input(question):
        return "INVALID_INPUT"

    messages = [
        SystemMessage(
            content=(
                "당신은 네트워크 트러블슈팅 Agent의 입력 라우팅 도구입니다. "
                "현재 사용자 입력만 보고 아래 세 가지 중 하나로 분류하세요.\n\n"
                "분류 기준:\n"
                "- VALID_INPUT: 네트워크 장애 질문이거나, 네트워크 장애 대화에서 이어질 수 있는 후속 응답입니다.\n"
                "  예: SSH 접속이 안 돼요, ping은 돼요, 방화벽 문제일까요, 여전히 안 됩니다, 그건 됩니다, Docker 컨테이너에서 인터넷이 안 돼요\n"
                "- GENERAL_CHAT: 네트워크 진단을 실행할 필요가 없는 일반 대화입니다.\n"
                "  예: 고마워, 해결됐어, 너 뭐 할 수 있어, 이 프로젝트 뭐야, 안녕\n"
                "- INVALID_INPUT: 의미 없는 반복 문자, 무작위 입력, 해석 불가능한 문자열입니다.\n"
                "  예: ㅁㅁㅁㅁㅁㅁㅁㅁㅁ, asdfasdfasdfasdf\n\n"
                "중요 규칙:\n"
                "- 이전 대화 문맥은 고려하지 마세요.\n"
                "- 현재 입력만 보고 판단하세요.\n"
                "- 사용자가 감사, 인사, 기능 질문을 하면 GENERAL_CHAT으로 분류하세요.\n"
                "- 네트워크 문제처럼 보이지만 준비된 유형에 딱 맞지 않아도 VALID_INPUT으로 분류하세요.\n"
                "- 반드시 아래 목록 중 하나만 출력하세요.\n"
                "- 설명 문장은 쓰지 마세요.\n\n"
                f"선택 가능한 유형: {INPUT_GUARD_TYPES}\n\n"
                "- 네트워크 키워드가 포함되어 있어도, 사용자의 실제 의도가 음식, 잡담, 감사, 인사, 일반 대화라면 GENERAL_CHAT으로 분류하세요.\n"
                "- 'DNS는 해결했는데 라면 먹고 싶다', 'SSH는 됐고 이제 밥 먹자'처럼 네트워크 문제 종료 후 다른 주제로 넘어가면 GENERAL_CHAT입니다.\n"
                "- 반대로 'DNS 서버 주소는 어디서 확인해요?', '그 명령어 결과는 어떻게 해석해요?', '포트가 열렸는지 어디서 봐요?'처럼 네트워크 설정 확인 방법이나 결과 해석을 묻는 질문은 VALID_INPUT입니다.\n"
                "- 짧은 후속 질문이라도 네트워크 점검 방법, 명령어, 설정 위치, 결과 해석을 묻고 있으면 VALID_INPUT입니다.\n"
                "- VALID_INPUT은 네트워크 장애 대상, 증상, 명령어 결과, 설정 확인 질문, 오류 메시지 중 하나 이상이 포함된 경우에만 선택하세요.\n"
                "- 네트워크 장애 정보가 없는 짧은 일반 대화, 장난성 문장, 무관한 질문은 VALID_INPUT으로 분류하지 마세요.\n"
                "- 예: '테스트', '아무 말', '오늘 뭐 먹지', '그냥 해본 말'처럼 네트워크 증상이나 점검 정보가 없으면 GENERAL_CHAT입니다.\n"
                "- 의미 없는 반복 문자나 해석하기 어려운 문자열은 INVALID_INPUT입니다.\n"
                "- 단, 표현이 자연스럽지 않더라도 네트워크 장애 대상이나 증상이 포함되어 있으면 VALID_INPUT입니다.\n"
                "- 예: '도커가 안 됩니다', '인터넷 연결이 안 됩니다', 'SSH 접속이 실패합니다', 'DNS가 동작하지 않습니다'는 VALID_INPUT입니다.\n"
                "- 네트워크 관련성이 애매하면 VALID_INPUT이 아니라 GENERAL_CHAT으로 분류하세요.\n"
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

    for guard_type in INPUT_GUARD_TYPES:
        if guard_type in result:
            return guard_type

    return "GENERAL_CHAT"

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
        chunk_size=700,
        chunk_overlap=100,
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
    
    documents = rag_store.similarity_search(question, k=2)

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


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    memory_context: str
    chat_history: List[dict]
    should_save_memory: bool
    memory_saved: bool
    graph_flow: List[str]
    problem_type: str
    recommended_commands: List[str]
    rag_result: str
    answer: str
    structured_result: Dict[str, Any]
    diagnosis_tool_result: str
    command_tool_result: List[str]
    input_guard_result: str
    input_status: str


def append_graph_flow(state: AgentState, node_name: str) -> List[str]:
    """
    실제 실행된 LangGraph 노드 흐름 기록
    """
    current_flow = state.get("graph_flow", ["START"])
    return current_flow + [node_name]


def memory_node(state: AgentState) -> dict:
    """
    session_id에 해당하는 이전 대화 이력을 불러오는 노드
    """
    session_id = state.get("session_id", "default")
    history = get_memory_history(session_id)
    memory_context = build_memory_context(history.messages)

    return{
        "session_id": session_id,
        "memory_context": memory_context,
        "chat_history": messages_to_json(history.messages),
        "graph_flow": append_graph_flow(state, "memory_node"),
    }


def input_guard_node(state: AgentState) -> dict:   #19 버그해결 로직
    """
    현재 질문이 Agent 진단에 적합한 입력인지 먼저 검사하는 노드
    Memory context는 사용 안 함
    """
    session_id = state.get("session_id", "default")

    # 명확한 ㅁ 반복 입력은 LLM 호출 없이 바로 차단
    if is_repeated_mieum_input(state["question"]):
        input_status = "INVALID_INPUT"
    else:
        model_call_limit_middleware(
            session_id=session_id,
            purpose="current input guard validation",
        )

        input_status = classify_current_input(state["question"])

    return {
        "input_status": input_status,
        "input_guard_result": input_status,
        "graph_flow": append_graph_flow(state, "input_guard_node"),
    }


def route_by_input_status(state: AgentState) -> str:
    """
    입력 검증 결과에 따라 다음 노드를 결정
    """
    if state["input_status"] == "VALID_INPUT":
        return "valid_input"

    if state["input_status"] == "INVALID_INPUT":
        return "invalid_input"

    return "general_chat"


def invalid_input_node(state: AgentState) -> dict:
    """
    의미 없는 입력인 경우 Agent 진단을 실행하지 않고 안내 메시지를 반환
    """
    structured_result = DiagnosisResult(
        problem_type="UNKNOWN_NETWORK_ISSUE",
        possible_causes=[
            "의미 없는 반복 입력 또는 해석하기 어려운 문자열입니다."
        ],
        recommended_commands=[],
        next_question="네트워크 장애 상황을 구체적인 문장으로 입력해주세요.",
        user_facing_answer="입력이 의미 없는 반복 문자열로 보여 네트워크 진단을 실행하지 않았습니다.",
    )

    answer = (
        "입력이 의미 없는 반복 문자열로 보여 네트워크 진단을 실행하지 않았습니다. "
        "SSH, DNS, DHCP, Gateway, Firewall 등 어떤 문제가 발생했는지 문장으로 입력해주세요."
    )

    return {
        "problem_type": "UNKNOWN_NETWORK_ISSUE",
        "recommended_commands": [],
        "command_tool_result": [],
        "rag_result": "",
        "answer": answer,
        "structured_result": structured_result.model_dump(),
        "should_save_memory": False,
        "graph_flow": append_graph_flow(state, "conditional_edge:invalid_input") + ["invalid_input_node"],
    }


async def general_chat_node(state: AgentState) -> dict:
    """
    네트워크 진단이 필요 없는 일반 대화에 자연스럽게 응답하는 노드
    """
    session_id = state.get("session_id", "default")

    model_call_limit_middleware(
        session_id=session_id,
        purpose="general chat response generation",
    )

    messages = [
        SystemMessage(
            content=(
                "당신은 Network Troubleshooting Agent입니다. "
                "사용자가 일반 대화를 하면 자연스럽게 응답하세요. "
                "단, 서비스의 중심은 네트워크 트러블슈팅임을 너무 딱딱하지 않게 알려주세요.\n\n"
                "응답 규칙:\n"
                "- UNKNOWN_NETWORK_ISSUE 같은 진단 문구를 말하지 마세요.\n"
                "- 사용자가 감사하면 짧고 자연스럽게 답하세요.\n"
                "- 사용자가 기능을 물으면 SSH, DNS, DHCP, Gateway, Firewall, Port 문제를 도와줄 수 있다고 설명하세요.\n"
                "- 네트워크 진단 Tool, RAG, 명령어 추천을 실행한 것처럼 말하지 마세요."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    response = await llm.ainvoke(messages)
    
    structured_result = DiagnosisResult(
        problem_type="GENERAL_CHAT",
        possible_causes=[],
        recommended_commands=[],
        next_question="네트워크 장애 상황이 있다면 증상을 알려주세요.",
        user_facing_answer=response.content,
    )

    return{
        "problem_type": "GENERAL_CHAT",
        "recommended_commands": [],
        "command_tool_result": [],
        "rag_result": "",
        "answer": response.content,
        "structured_result": structured_result.model_dump(),
        "should_save_memory": False,
        "graph_flow": append_graph_flow(state, "conditional_edge:general_chat") + ["general_chat_node"],
    }


def diagnose_node(state: AgentState) -> dict:
    """
    사용자 질문과 이전 대화 이력을 기반으로 네트워크 장애 유형을 분류하는 노드
    """
    session_id = state.get("session_id", "default")

    diagnosis_input = (
        f"이전 대화 이력:\n{state.get('memory_context','')}\n\n"
        f"현재 질문:\n{state['question']}"
    )

    tool_call_limit_middleware(
        session_id=session_id,
        tool_name="network_diagnosis_tool",
        tool_args={"question": state["question"]},
    )

    model_call_limit_middleware(
        session_id=session_id,
        purpose="network diagnosis classification",
    )

    diagnosis_type = network_diagnosis_tool.invoke(
        {"question": diagnosis_input}
    )
    return{
        "problem_type": diagnosis_type,
        "diagnosis_tool_result": diagnosis_type,
        "graph_flow": append_graph_flow(state, "diagnose_node"),
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
    사용자 질문과 이전 대화 이력을 함께 사용하여 RAG 문서를 검색하는 노드
    """
    session_id = state.get("session_id", "default")

    rag_query =(
        f"이전 대화 이력:\n{state.get('memory_context', '')}\n\n"
        f"현재 질문:\n{state['question']}\n\n"
        f"진단 유형:\n{state['problem_type']}"
    )

    tool_call_limit_middleware(
        session_id=session_id,
        tool_name="rag_search_tool",
        tool_args={"question": state["question"]},
    )

    rag_result = rag_search_tool.invoke(
        {"question": rag_query}
    )
    return{
        "rag_result": rag_result,
        "graph_flow": append_graph_flow(state, "conditional_edge:known_problem") + ["rag_node"],
    }


def command_node(state: AgentState) -> dict:
    """
    장애 유형에 따라 점검 명령어를 추천하는 노드
    """
    session_id = state.get("session_id", "default")

    tool_call_limit_middleware(
        session_id=session_id,
        tool_name="command_recommendation_tool",
        tool_args={"problem_type": state["problem_type"]},
    )

    recommended_commands = command_recommendation_tool.invoke(
        {"problem_type": state["problem_type"]}
    )
    return{
        "recommended_commands": recommended_commands,
        "command_tool_result": recommended_commands,
        "graph_flow": append_graph_flow(state, "command_node"),
    }


async def generate_answer_node(state: AgentState) -> dict:
    """
    진단 결과, RAG 검색 결과, 추천 명령어를 바탕으로 최종 답변을 생성하는 노드
    """
    session_id = state.get("session_id", "default")

    model_call_limit_middleware(
        session_id=session_id,
        purpose="final troubleshooting answer generation",
    )

    messages = [
        SystemMessage(
            content=(
                "당신은 네트워크 트러블슈팅을 도와주는 AI Assistant입니다. "
                "사용자의 네트워크 장애 상황을 듣고 가능한 원인과 다음 확인 단계를 "
                "쉽고 간단하게 설명하세요. "
                "Memory를 사용하여 이전 대화 이력을 참고하고, "
                "LangGraph 기반 진단 흐름과 RAG 검색 결과를 함께 활용합니다.\n\n"
                f"이전 대화 이력은 다음과 같습니다:\n{state.get('memory_context', '이전 대화 이력이 없습니다.')}\n\n"
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
                "- 이전 대화에서 이미 확인된 정보는 다시 묻지 마세요.\n"
                "- 사용자가 ping이 된다고 말했거나 방화벽 가능성을 물어보면, 서버 IP보다 SSH 서비스 상태, 포트 리스닝 여부, 방화벽 허용 여부를 우선 질문하세요.\n"
                "- user_facing_answer에는 사용자가 실제로 읽을 자연스러운 답변을 작성하세요.\n"
                "- user_facing_answer에서는 problem_type을 기계적으로 반복하지 말고, 상황을 해석해서 설명하세요.\n"
                "- 사용자가 입력한 IP, 포트, OS, 오류 메시지, 이미 확인한 결과가 있다면 답변에 반영하세요.\n"
                "- 준비된 유형에 딱 맞지 않는 네트워크 문제라도 가능한 확인 순서를 제안하세요.\n"
                "- 확실하지 않은 원인은 단정하지 말고 가능성으로 표현하세요.\n"
                "- user_facing_answer 안에도 recommended_commands 중 핵심 명령어 2~4개를 직접 포함하세요.\n"
                "- '명령어를 추천드립니다'라고만 말하지 말고, 사용자가 바로 실행할 수 있게 명령어를 실제로 적으세요.\n"
                "- 답변은 상황 해석 → 가능한 원인 → 확인 명령어 → 결과를 알려달라는 요청 순서로 작성하세요.\n"
                "- 명령어를 나열할 때 각 명령어가 무엇을 확인하는지도 짧게 설명하세요.\n"
                "- JSON 밖에 일반 문장, 설명, 번호 목록을 절대 쓰지 마세요.\n"
                "- 응답의 첫 글자는 반드시 { 이고 마지막 글자는 반드시 } 여야 합니다.\n"
                "- user_facing_answer 안에 자연스러운 설명을 넣고, JSON 바깥에는 아무것도 쓰지 마세요.\n"
            )
        ),
        HumanMessage(content=state["question"]),
    ]

    response = await llm.ainvoke(messages)

    try:
        parsed_result = parser.parse(response.content)
        parsed_result.problem_type = state["problem_type"]
        parsed_result.recommended_commands = state["recommended_commands"]

    except Exception as e:
        add_middleware_log(
            session_id=session_id,
            middleware="outputParserFallback",
            level="warn",
            message="OutputParser 파싱 실패, fallback structured result 사용",
            detail=str(e),
        )

        fallback_answer = response.content.strip()

        if not fallback_answer:
            fallback_answer = (
                "답변 생성 중 구조화 응답을 만들지 못했습니다. "
                "아래 점검 명령어를 실행한 뒤 결과를 알려주세요."
            )

        parsed_result = DiagnosisResult(
            problem_type=state["problem_type"],
            possible_causes=[
                "LLM 응답이 JSON 형식을 지키지 않아 구조화 파싱에 실패했습니다.",
                "현재 진단 유형과 추천 명령어를 기준으로 후속 점검이 필요합니다.",
            ],
            recommended_commands=state["recommended_commands"],
            next_question="위 명령어 실행 결과를 알려주세요.",
            user_facing_answer=fallback_answer,
        )

    answer = parsed_result.user_facing_answer

    return {
        "answer": answer,
        "structured_result": parsed_result.model_dump(),
        "should_save_memory": True,
        "graph_flow": append_graph_flow(state, "generate_answer_node"),
    }


async def clarification_node(state: AgentState) -> dict:
    """
    준비된 장애 유형에 딱 맞지 않는 네트워크 질문에 대해 범용 트러블슈팅 답변을 생성하는 노드
    """
    session_id = state.get("session_id", "default")

    tool_call_limit_middleware(
        session_id=session_id,
        tool_name="command_recommendation_tool",
        tool_args={"problem_type": "UNKNOWN_NETWORK_ISSUE"},
    )

    base_commands = command_recommendation_tool.invoke(
        {"problem_type": "UNKNOWN_NETWORK_ISSUE"}
    )

    model_call_limit_middleware(
        session_id=session_id,
        purpose="adaptive unknown network troubleshooting answer generation",
    )

    messages = [
        SystemMessage(
            content=(
                "당신은 네트워크 트러블슈팅을 도와주는 AI Assistant입니다. "
                "사용자의 질문이 준비된 장애 유형에 딱 맞지 않더라도, "
                "네트워크 관련 문제라면 일반적인 트러블슈팅 절차로 대응하세요.\n\n"
                f"이전 대화 이력:\n{state.get('memory_context', '이전 대화 이력이 없습니다.')}\n\n"
                "현재 진단 유형은 UNKNOWN_NETWORK_ISSUE입니다. "
                "이는 네트워크 문제가 아니라는 뜻이 아니라, 준비된 세부 유형으로 특정하기 어렵다는 뜻입니다.\n\n"
                f"기본 점검 명령어 후보는 다음과 같습니다:\n{base_commands}\n\n"
                "응답은 반드시 아래 Pydantic JSON 형식 지침을 따르세요.\n"
                f"{parser.get_format_instructions()}\n\n"
                "작성 규칙:\n"
                "- 반드시 JSON 형식으로만 답변하세요.\n"
                "- 마크다운 코드블록은 사용하지 마세요.\n"
                "- problem_type은 반드시 UNKNOWN_NETWORK_ISSUE로 작성하세요.\n"
                "- user_facing_answer에는 사용자가 실제로 읽을 자연스러운 답변을 작성하세요.\n"
                "- 사용자가 말한 환경, 예를 들어 Docker, VM, VPN, 프록시, 특정 IP, 포트, OS, 오류 메시지가 있으면 반드시 반영하세요.\n"
                "- 준비된 유형에 딱 맞지 않아도 가능한 원인 후보를 제시하세요.\n"
                "- 확실하지 않은 원인은 단정하지 말고 가능성으로 표현하세요.\n"
                "- recommended_commands에는 기본 점검 명령어를 포함하되, 질문 맥락에 맞는 추가 명령어가 있으면 함께 제안하세요.\n"
                "- next_question에는 추가 진단을 위해 가장 필요한 정보를 하나 물어보세요.\n"
                "- Docker 또는 컨테이너 문제라면 docker exec, ip route, cat /etc/resolv.conf, ping 8.8.8.8, nslookup google.com 같은 컨테이너 내부 확인 명령어를 제안하세요.\n"
                "- VM 문제라면 게스트 OS의 IP, NAT/Bridged 설정, 게이트웨이, DNS 설정 확인을 제안하세요.\n"
                "- VPN 문제라면 VPN 연결 후 라우팅 테이블, DNS suffix, split tunneling, 사내 대역 route 확인을 제안하세요.\n"
                "- user_facing_answer 안에도 사용자가 바로 따라 할 수 있는 점검 명령어를 3개 이상 포함하세요.\n"
                "- 당신이 직접 명령어를 실행할 수 있다고 말하지 마세요.\n"
                "- '실행해 보겠습니다'가 아니라 '실행해보세요', '결과를 알려주세요'라고 말하세요.\n"
                "- Docker 또는 컨테이너 문제라면 docker exec, ip route, cat /etc/resolv.conf, ping 8.8.8.8, nslookup google.com 같은 컨테이너 내부 확인 명령어를 답변에 직접 포함하세요.\n"
                "- 답변은 원인 후보 → 확인 명령어 → 결과 해석 → 다음 질문 순서로 작성하세요.\n"
                "- user_facing_answer에는 명령어만 나열하지 말고, 각 결과를 어떻게 해석해야 하는지도 함께 설명하세요.\n"
                "- ping 명령어는 무한 실행되지 않도록 가능하면 ping -c 4 형태로 제안하세요.\n"
                "- Docker 문제에서는 host 인터넷 문제와 container 내부 문제를 구분하는 기준을 설명하세요.\n"
                "- 마지막에는 컨테이너 ID뿐 아니라 실행 결과도 함께 알려달라고 요청하세요.\n"
            )
        ),
        HumanMessage(content=state["question"]),
    ]

    try:
        response = await llm.ainvoke(messages)
        parsed_result = parser.parse(response.content)

        parsed_result.problem_type = "UNKNOWN_NETWORK_ISSUE"

        if not parsed_result.recommended_commands:
            parsed_result.recommended_commands = base_commands

    except Exception:
        parsed_result = DiagnosisResult(
            problem_type="UNKNOWN_NETWORK_ISSUE",
            possible_causes=[
                "준비된 장애 유형에 딱 맞지 않거나 질문 정보가 부족하여 원인을 특정하기 어렵습니다."
            ],
            recommended_commands=base_commands,
            next_question="사용 중인 환경, 대상 IP/도메인, 포트 번호, 오류 메시지, 이미 확인한 명령어 결과를 알려주세요.",
            user_facing_answer=(
                "준비된 장애 유형에 딱 맞지는 않지만 네트워크 문제일 가능성이 있습니다. "
                "먼저 IP 연결, DNS, 라우팅, 포트 차단 여부를 차례로 확인해보는 게 좋습니다. "
                "사용 중인 환경과 오류 메시지, 이미 확인한 결과를 알려주면 더 구체적으로 좁혀볼 수 있습니다."
            ),
        )

    return {
        "problem_type": "UNKNOWN_NETWORK_ISSUE",
        "recommended_commands": parsed_result.recommended_commands,
        "command_tool_result": parsed_result.recommended_commands,
        "rag_result": "",
        "answer": parsed_result.user_facing_answer,
        "structured_result": parsed_result.model_dump(),
        "should_save_memory": True,
        "graph_flow": append_graph_flow(state, "conditional_edge:clarification") + ["clarification_node"],
    }


def save_memory_node(state: AgentState) -> dict:
    """
    현재 사용자 질문과 최종 답변을 session memory에 저장하는 노드
    """
    session_id = state.get("session_id", "default")
    history = get_memory_history(session_id)

    if not state.get("should_save_memory", True):
        return{
            "chat_history": messages_to_json(history.messages),
            "memory_saved": False,
            "graph_flow": append_graph_flow(state, "save_memory_node") + ["END"],
        }

    history.add_message(HumanMessage(content=state["question"]))
    history.add_message(AIMessage(content=state.get("answer", "")))

    return{
        "chat_history": messages_to_json(history.messages),
        "memory_saved": True,
        "graph_flow": append_graph_flow(state, "save_memory_node") + ["END"],
    }


workflow = StateGraph(AgentState)

workflow.add_node("memory_node", memory_node)
workflow.add_node("input_guard_node", input_guard_node)
workflow.add_node("general_chat_node", general_chat_node)
workflow.add_node("diagnose_node", diagnose_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("command_node", command_node)
workflow.add_node("generate_answer_node", generate_answer_node)
workflow.add_node("clarification_node", clarification_node)
workflow.add_node("invalid_input_node", invalid_input_node)
workflow.add_node("save_memory_node", save_memory_node)

workflow.add_edge(START, "memory_node")
workflow.add_edge("memory_node", "input_guard_node")

workflow.add_conditional_edges( #19 버그해결
    "input_guard_node",
    route_by_input_status,
    {
        "valid_input": "diagnose_node",
        "general_chat": "general_chat_node",
        "invalid_input": "invalid_input_node",
    },
)

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
workflow.add_edge("generate_answer_node", "save_memory_node")
workflow.add_edge("clarification_node", "save_memory_node")
workflow.add_edge("invalid_input_node", "save_memory_node")
workflow.add_edge("general_chat_node", "save_memory_node")
workflow.add_edge("save_memory_node", END)

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
        request_logging_middleware(
            session_id=req.session_id,
            question=req.question,
        )
        final_state = await agent_graph.ainvoke(
            {
                "question": req.question,
                "session_id": req.session_id,
            }
        )
        agent_finish_logging_middleware(
            session_id=req.session_id,
            final_state=final_state,
        )
        return{
            "success": True,
            "question": req.question,
            "session_id": req.session_id,
            "memory_used": True,
            "memory_saved": final_state.get("memory_saved", False),
            "chat_history": final_state.get("chat_history", []),
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
            "graph_flow": final_state.get("graph_flow", []),
            "middleware_used": True,
            "middleware_logs": get_recent_middleware_logs(req.session_id),
            "middleware_stats": {
                "model_call_count": get_middleware_session(req.session_id)["model_call_count"],
                "tool_call_count": get_middleware_session(req.session_id)["tool_call_count"],
            },
            "model": "gpt-4o-mini",
        }
        
    
    except Exception as e:
        agent_error_logging_middleware(
            session_id=req.session_id,
            error=e,
        )

        return {
            "success": False,
            "error": str(e),
            "session_id": req.session_id,
            "middleware_used": True,
            "middleware_logs": get_recent_middleware_logs(req.session_id),
        }


class ResetMemoryRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=40)

@app.post("/api/memory/reset")
async def reset_memory(req: ResetMemoryRequest):
    memory_stores.pop(req.session_id, None)

    return{
        "success": True,
        "message": f"{req.session_id} 세션 메모리가 초기화되었습니다."
    }


@app.get("/api/memory/history")
async def get_memory_history_api(
    session_id: str = Query(default="default"),
):
    history = get_memory_history(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "chat_history": messages_to_json(history.messages),
    }


class ResetMiddlewareRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=40)


@app.get("/api/middleware/logs")
async def get_middleware_logs(
    session_id: str = Query(default="default"),
    limit: int = Query(default=20),
):
    session = get_middleware_session(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "logs": get_recent_middleware_logs(session_id, limit),
        "stats": {
            "model_call_count": session["model_call_count"],
            "tool_call_count": session["tool_call_count"],
        },
        "config": session["config"],
    }


@app.post("/api/middleware/reset")
async def reset_middleware(req: ResetMiddlewareRequest):
    middleware_sessions.pop(req.session_id, None)

    return {
        "success": True,
        "message": f"{req.session_id} 세션 middleware 로그가 초기화되었습니다.",
    }



class MiddlewareConfigRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=40)
    model_call_limit: Optional[int] = None
    tool_call_limit: Optional[int] = None
    log_retention: Optional[int] = None



@app.post("/api/middleware/config")
async def update_middleware_config(req: MiddlewareConfigRequest):
    session = get_middleware_session(req.session_id)

    if req.model_call_limit is not None:
        session["config"]["model_call_limit"] = req.model_call_limit

    if req.tool_call_limit is not None:
        session["config"]["tool_call_limit"] = req.tool_call_limit

    if req.log_retention is not None:
        session["config"]["log_retention"] = req.log_retention

    return {
        "success": True,
        "session_id": req.session_id,
        "config": session["config"],
    }


app.mount(
    "/public",
    StaticFiles(directory=PUBLIC_DIR, check_dir=False),
    name="public",
)

@app.get("/")
async def index():
    return FileResponse(PUBLIC_DIR / "index.html")