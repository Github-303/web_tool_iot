from langchain_openai import ChatOpenAI

def load_llm(model_name):
    """Load large language model

    Args:
        model_name (str): Name of the model to load

    Returns:
        BaseChatModel: A chat model instance
    """
    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=1000)
    elif model_name == "gpt-3.5-turbo":
        return ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=1000)
    else:
        raise ValueError(f"Invalid model name: {model_name}")