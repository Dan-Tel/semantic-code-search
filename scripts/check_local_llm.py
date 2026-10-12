from langchain_ollama import ChatOllama


def main():
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    response = llm.invoke(
        "Explain briefly what an HTTP Authorization header is."
    )

    print(response.content)


if __name__ == "__main__":
    main()