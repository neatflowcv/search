# LangGraph Patterns (2026)

LangGraph를 사용한 현대적인 AI 에이전트 구축 패턴 정리.

## 핵심 개념

LangGraph는 LLM 워크플로우를 위한 **결정론적 실행 엔진**으로, 상태가 흐르고
결정이 이루어지며 로직이 명시적인 그래프로 워크플로우를 설계할 수 있다.

## StateGraph와 TypedDict

상태는 모든 노드에서 접근 가능한 공유 메모리로 작동한다.

```python
from typing import Annotated, TypedDict, Literal
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """에이전트 상태 정의"""
    # 원시 데이터 저장 (포맷된 텍스트가 아님)
    query: str

    # add reducer로 메시지 누적
    messages: Annotated[list[BaseMessage], add]

    # 검색 결과
    search_results: list[dict]

    # 최종 응답
    response: str | None
```

### 상태 설계 원칙

1. 단계 간 유지되거나 재계산 비용이 높은 데이터 포함
2. 기존 데이터에서 계산 가능한 파생 정보 제외
3. 포맷된 텍스트 대신 원시 데이터 저장
4. 노드 내부에서 필요시 프롬프트 포맷팅

## 노드 유형

### LLM 노드

분류, 분석, 텍스트 생성 담당.

```python
async def classify_node(state: AgentState) -> dict:
    """쿼리 분류"""
    llm = get_llm_client()
    result = await llm.ainvoke([
        {"role": "system", "content": "Classify the query..."},
        {"role": "user", "content": state["query"]},
    ])
    return {"classification": result.content}
```

### 데이터 노드

외부 시스템에서 정보 검색.

```python
async def search_node(state: AgentState) -> dict:
    """외부 검색 실행"""
    client = get_search_client()
    results = await client.search(state["query"])
    return {"search_results": results}
```

### 액션 노드

이메일 전송, 티켓 생성 등 외부 작업 수행.

## Command 기반 라우팅

노드가 상태 업데이트와 라우팅을 동시에 처리.

```python
from langgraph.graph import Command

async def research_node(state: AgentState) -> Command[Literal["search", "respond"]]:
    """연구 노드 - 다음 단계 결정"""
    llm = get_llm_client()
    response = await llm.ainvoke(state["messages"])

    # 도구 호출 파싱
    tool_calls = parse_tool_calls(response.content)

    if "done" in [tc["name"] for tc in tool_calls]:
        return Command(
            update={"is_complete": True},
            goto="respond"
        )

    return Command(
        update={"pending_searches": tool_calls},
        goto="search"
    )
```

## RetryPolicy

일시적 실패에 대한 복원력 제공.

```python
from langgraph.types import RetryPolicy

graph.add_node(
    "search",
    search_node,
    retry=RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        backoff_factor=2.0,
        jitter=True,
    )
)
```

### 오류 처리 전략

| 오류 유형 | 처리 방법 |
|----------|----------|
| 일시적 (네트워크, 속도 제한) | RetryPolicy 적용 |
| LLM 복구 가능 | 상태에 오류 저장, LLM으로 루프백 |
| 사용자 수정 필요 | interrupt()로 일시 중지 |
| 예기치 않은 오류 | 버블링 허용 |

## 그래프 구성

```python
from langgraph.graph import StateGraph, START, END

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("research", research_node)
    graph.add_node("search", search_node)
    graph.add_node("respond", respond_node)

    # 엣지 정의
    graph.add_edge(START, "research")
    graph.add_edge("search", "research")  # 루프백
    graph.add_edge("respond", END)

    return graph.compile()
```

## Human-in-the-Loop

`interrupt()` 함수로 그래프 일시 중지.

```python
from langgraph.types import interrupt, Command

def human_review(state) -> Command[Literal["send", END]]:
    decision = interrupt({
        "draft": state["draft"],
        "action": "Please review and approve"
    })

    if decision.get("approved"):
        return Command(goto="send")
    return Command(goto=END)
```

## 체크포인트와 지속성

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 실행
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(initial_state, config)

# 나중에 동일 thread_id로 재개 가능
```

## 로컬 LLM 연동

OpenAI 호환 API를 제공하는 로컬 서버와 연동.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:7000/v1",
    api_key="local-key",
    model="jan-v3-4b",
    temperature=0.1,
)
```

## 참고 자료

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
