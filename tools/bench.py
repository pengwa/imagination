import traceback

# from ai_search_engine.search_engine.services.llm_services import LLMServiceProxy
# from ai_search_engine.search_engine.domain.model import Model
# from ai_search_engine.search_engine.services.rag import EXTENDED_QUERIES_PROMPT
import time
import os

query = "Analysis of the impact of the COVID-19 pandemic on the global economy"
history = ""

user_prompt = "Query: \n{query}\n\nPrevious chat history (if any): \n{history}".format(
    query=query, history=history
)

EXTENDED_QUERIES_PROMPT1 = """
Given a user's query and any available chat history, identify the main intent of the query and rewrite it into three distinct, clear, and complete search queries. Each query should independently express the user's intent, incorporating relevant details such as time, location, persons, or events as mentioned by the user. Ensure the queries are detailed and specific, using appropriate keywords to help retrieve comprehensive and targeted information.

Instructions:

Do not alter or add any entities (like time, location, or person) mentioned in the query.
Do not remove any information from the original query.
Output only one query per line, and avoid adding any extra information.
Note:

If the query involves math problems, code issues, reading comprehension, text summarization, or translation, do not output anything.
If the query is unclear or cannot be understood, do not output anything.
The current time is {current_time}. 
"""
# The current time is {current_time}.

import ollama


os.environ["OPENAI_API_ENDPOINT"] = "https://search-engine-aoai-usw3.openai.azure.com"

# llm_proxy = LLMServiceProxy()
# model = Model.GPT_4O

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
p = EXTENDED_QUERIES_PROMPT1.format(current_time=current_time)

for i in range(1):
    # time.sleep(20)
    t_start = time.time()
    try:

        print("len of system prompt: ", len(p))
        chat_complete_kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": p,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "max_tokens": 1024,
            "temperature": 0.5,
        }

        from ollama import Client

        user_prompt = "Why is the sky blue?"
        model_name = "llama2"
        # model_name="phi3:mini"
        # model_name="phi3:medium"
        client = Client(host="http://localhost:11434")
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": p,
                },
                {
                    "role": "user",
                    "content": (
                        p + "\n" + user_prompt if "phi3" in model_name else user_prompt
                    ),
                },
            ],
        )

        t_end = None

        if t_end is None:
            t_end = time.time()

        print(f"Time taken: {t_end - t_start} seconds, >>>> ", response)
    except Exception as e:
        print(f"rewrite llm encountered error: {e}\n{traceback.format_exc()}")
