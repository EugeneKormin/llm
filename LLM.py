from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import OpenAI
import httpx

from ConfigReader import OPENAI_API_TOKEN


class LLM(object):
    def __init__(self):
        self.__OPEN_AI_KEY: str = OPENAI_API_TOKEN
        self.__chain: RunnableSerializable[dict, str]

    def __get_template(self) -> RunnableSerializable:
        return PromptTemplate.from_template("task: {task}")

    def get_response(self, TASK: str) -> str:
        llm: OpenAI = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            openai_api_key=self.__OPEN_AI_KEY,
            max_tokens=-1,
        )

        template = self.__get_template()

        main_llm_chain: RunnableSerializable[dict, str] = template | llm
        RESPONSE: str = main_llm_chain.invoke(input={
            "task": TASK
        })

        return RESPONSE
