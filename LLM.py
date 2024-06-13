from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import OpenAI

from ConfigReader import OPENAI_API_TOKEN


class LLM(object):
    def __init__(self):
        self.__OPEN_AI_KEY: str = OPENAI_API_TOKEN
        self.__chain: RunnableSerializable[dict, str]

    def __get_updated_chain(self, TASK: str) -> None:
        main_propmt: PromptTemplate = PromptTemplate.from_template(TASK)
        llm: OpenAI = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            openai_api_key=self.__OPEN_AI_KEY,
            max_tokens=-1,
        )
        self.__llm_chain: RunnableSerializable[dict, str] = main_propmt | llm

    def get_response(self, TEXT: str) -> str:
        self.__get_updated_chain(TASK=TEXT)
        return self.__llm_chain.invoke(input={
            "context": "To relocate you must have lots of docs."
        })
