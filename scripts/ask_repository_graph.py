from time import perf_counter
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from semantic_code_search.search import load_repository_search

from scripts.ask_repository import (
    ROOT,
    MODEL_NAME,
    retrieve_context,
    generate_answer,
    print_results,
    save_log,
)


class SearchState(TypedDict, total=False):
    query: str
    search_query: str
    results: list[dict]
    context: str
    answer: str


def create_graph(search, llm):
    def rewrite_node(state):
        response = llm.invoke([
            (
                "system",
                "Convert the question into a short English description "
                "of the code operation to search for. "
                "Start with an action verb. "
                "Keep important technical terms and the intended action. "
                "Remove phrases such as 'where', 'in the code', "
                "'how does it work', and requests for explanations. "
                "Do not invent implementation details or function names. "
                "Output only the search phrase.",
            ),
            (
                "human",
                "Where are expired cache entries cleaned up, "
                "and how does that work?",
            ),
            ("ai", "remove expired cache entries"),
            (
                "human",
                "How does the application read settings from a JSON file?",
            ),
            ("ai", "read application settings from JSON file"),
            ("human", state["query"]),
        ])

        search_query = response.content.strip()

        if not search_query:
            search_query = state["query"]

        print("\nSearch query:", search_query)

        return {"search_query": search_query}

    def retrieve_node(state):
        retrieval = retrieve_context(
            search,
            state["search_query"],
        )

        print_results(retrieval["results"])

        return {
            "results": retrieval["results"],
            "context": retrieval["context"],
        }

    def answer_node(state):
        # Отвечаем на исходный вопрос пользователя.
        answer = generate_answer(
            llm,
            state["query"],
            state["context"],
        )

        return {"answer": answer}

    builder = StateGraph(SearchState)

    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


def main():
    search, _ = load_repository_search(ROOT)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url="http://localhost:11434",
        temperature=0,
        reasoning=False,
    )

    graph = create_graph(search, llm)

    while True:
        try:
            query = input(
                "\nQuestion (or 'exit'): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() == "exit":
            break

        if not query:
            continue

        started_at = perf_counter()

        result = graph.invoke({"query": query})

        elapsed_seconds = perf_counter() - started_at

        print("\nAnswer:")
        print(result["answer"])

        save_log(
            query=query,
            retrieval={
                "results": result["results"],
                "context": result["context"],
            },
            answer=result["answer"],
            elapsed_seconds=elapsed_seconds,
            search_query=result["search_query"],
        )

        print(
            f"\nSaved locally | "
            f"{elapsed_seconds:.2f} seconds"
        )


if __name__ == "__main__":
    main()