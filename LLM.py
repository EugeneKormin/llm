from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import tiktoken
import os
import json
from ConfigReader import OPENAI_API_TOKEN


class LLM(object):
    def __init__(self):
        self.__OPEN_AI_KEY: str = OPENAI_API_TOKEN
        self.__chain: RunnableSerializable[dict, str]
        self.__llm: OpenAI = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            openai_api_key=self.__OPEN_AI_KEY,
            max_tokens=-1,
        )

    def get_response_instagram(self, TASK: str) -> str:
        template = PromptTemplate.from_template("task: {task}")

        main_llm_chain: RunnableSerializable[dict, str] = template | self.__llm
        RESPONSE: str = main_llm_chain.invoke(input={
            "task": TASK
        })

        return RESPONSE

    def get_tokens(self, TEXT: str):
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return enc.encode(TEXT)

    def get_embeddings(self, TEXT: str):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_result = embeddings.embed_query(TEXT)
        return query_result

    def get_response_warhammer(self, TASK: str, PAGE_NUM: str, CONTEXT: str) -> str:
        template = PromptTemplate.from_template(f"task: {TASK}\npage_num: {PAGE_NUM}\ncontext: {CONTEXT}")

        main_llm_chain: RunnableSerializable[dict, str] = template | self.__llm
        RESPONSE: str = main_llm_chain.invoke(input={
            "page_num": PAGE_NUM, "context": CONTEXT
        })

        return RESPONSE
