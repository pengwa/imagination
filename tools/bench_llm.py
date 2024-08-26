import traceback
import time
import os
import numpy as np

os.environ["OPENAI_API_ENDPOINT"] = "https://search-engine-aoai-usw3.openai.azure.com"
os.environ["SEMANTIC_DOC_ENDPOINT"] = (
    ""
)
os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = ""
os.environ["GOOGLE_SEARCH_API_KEY"] = ""
os.environ["GOOGLE_SEARCH_ENGINE_ID"] = ""
from ai_search_engine.search_engine.services.llm_services import LLMServiceProxy
from ai_search_engine.search_engine.domain.model import Model

# from ai_search_engine.search_engine.services.rag import EXTENDED_QUERIES_PROMPT
from ollama import Client

query = "Analysis of the impact of the COVID-19 pandemic on the global economy"
history = ""

user_prompt = "Query: \n{query}\n\nPrevious chat history (if any): \n{history}".format(
    query=query, history=history
)

# 934
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
# 735
EXTENDED_QUERIES_PROMPT2 = """
Given a user's query and chat history, identify the query's main intent and rephrase it into three clear and complete search queries. Each should fully express the user's intent, including any specific details like time, location, or persons mentioned. Ensure the queries are detailed, using relevant keywords for targeted information retrieval.

Instructions:

Do not add or change any entities mentioned in the query.
Do not remove any information from the original query.
Output one query per line, with no extra information.
Note:

For math problems, code issues, reading comprehension, text summarization, or translation, do not output anything.
If the query is unclear, do not output anything.
The current time is {current_time}."""

# 685
EXTENDED_QUERIES_PROMPT3 = """
Identify the main intent of a user's query using chat history, then rewrite it into three distinct, clear search queries. Each query should fully express the user's intent, including details like time, location, or persons mentioned, and use relevant keywords for targeted information retrieval.

Instructions:

Do not add or change any entities mentioned in the query.
Do not remove any information from the original query.
Output one query per line, with no extra information.
Note:

For math problems, code issues, reading comprehension, text summarization, or translation, do not output anything.
If the query is unclear, do not output anything.
The current time is {current_time}.
"""
# The current time is {current_time}.

# 593
EXTENDED_QUERIES_PROMPT4 = """
Identify the main intent of a user's query using chat history, then rewrite it into three clear search queries. Each query should fully express the user's intent, including details like time, location, or persons mentioned, and use relevant keywords.

Instructions:

Do not add, change, or remove any details from the original query.
Output one query per line with no extra information.
Note:

For math problems, code issues, reading comprehension, text summarization, or translation, do not output anything.
If the query is unclear, do not output anything.
The current time is {current_time}.
"""

# 572
EXTENDED_QUERIES_PROMPT5 = """
Rewrite a user's query into three distinct, detailed search queries that reflect the main intent and include any mentioned entities. Each query should be clear, specific, and use appropriate keywords for precise results.

Instructions:

Do not add or remove entities (like time, location, or persons) mentioned in the query.
Output one query per line without any extra information.
If the query involves math, code, reading comprehension, summarization, or translation, do not output anything.
If the query is unclear, do not output anything.
Current time: {current_time}.
"""

import ollama

llm_proxy = LLMServiceProxy()
model_name = Model.PHI3_MEDIUM
model_name = Model.GPT_4O
model_name = "ollama:phi3:mini"
model_name = "ollama:phi3:medium"
# model_name = "ollama:phi3:3.8b-mini-128k-instruct-q4_1"
# model_name = "ollama:phi3:3.8b-mini-128k-instruct-q4_0"
model_name = "ollama:phi3:3.8b-mini-4k-instruct-q4_0"
model_name = "ollama:phi3:3.8b-mini-4k-instruct-q4_1"
model_name = "ollama:phi3:3.8b-mini-4k-instruct-q4_K_S"
model_name = "ollama:phi3:3.8b-mini-4k-instruct-q4_K_M"
model_name = "ollama:phi3:3.8b-mini-4k-instruct-q8_0"

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

p = EXTENDED_QUERIES_PROMPT1.format(current_time=current_time)
print("wrapping up: len of system prompt: ", len(p))
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

client = Client(host="http://localhost:11434")
if "ollama:" in model_name:
    response = client.chat(
        model=model_name.replace("ollama:", ""),
        messages=[
            {
                "role": "system",
                "content": p,
            },
            {
                "role": "user",
                "content": (p + "\n" + user_prompt if "phi3" in model_name else user_prompt),
            },
        ],
    )
else:
    llm_response, response_iterator = llm_proxy.chat_completions_create(
        model_name=model_name, **chat_complete_kwargs
    )


for j in range(5):
    t_scope_start = time.time()
    collected_times = []
    for i in range(20):
        if j == 0:
            p = EXTENDED_QUERIES_PROMPT1.format(current_time=current_time)
        elif j == 1:
            p = EXTENDED_QUERIES_PROMPT2.format(current_time=current_time)
        elif j == 2:
            p = EXTENDED_QUERIES_PROMPT3.format(current_time=current_time)
        elif j == 3:
            p = EXTENDED_QUERIES_PROMPT4.format(current_time=current_time)
        else:
            p = EXTENDED_QUERIES_PROMPT5.format(current_time=current_time)
        # time.sleep(20)
        t_start = time.time()
        try:

            # print("len of system prompt: ", len(p))
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
            t_end = None

            if "ollama:" in model_name:
                response = client.chat(
                    model=model_name.replace("ollama:", ""),
                    messages=[
                        {
                            "role": "system",
                            "content": p,
                        },
                        {
                            "role": "user",
                            "content": (
                                p + "\n" + user_prompt
                                if "phi3" in model_name
                                else user_prompt
                            ),
                        },
                    ],
                )
            else:
                llm_response, response_iterator = llm_proxy.chat_completions_create(
                    model_name=model_name, **chat_complete_kwargs
                )

                for q in response_iterator(llm_response):
                    t_end = time.time()
                    break

            if t_end is None:
                t_end = time.time()

            # print(f"Time taken: {t_end - t_start} seconds, >>>> ")
            collected_times.append(t_end - t_start)

            # Replace double quotes with single quotes in the query to avoid exact search.
            # queries = [
            #     q.replace('"', "'").strip()
            #     for q in response_iterator(llm_response)
            # ]

            # return [q for q in queries if q]
        except Exception as e:
            print(f"rewrite llm encountered error: {e}\n{traceback.format_exc()}")
            # return [query]
    t_scope_end = time.time()
    print(
        f"L{len(p)}: average: {(t_scope_end-t_scope_start)/len(collected_times)} \nTime taken for scope {j}: {t_scope_end - t_scope_start} seconds, >>>>  length: {len(p)}, , std: {np.std(collected_times)},"
        f"max: {max(collected_times)}, min: {min(collected_times)}, collected_times: {collected_times}, \nhist: {np.histogram(collected_times)}"
    )
