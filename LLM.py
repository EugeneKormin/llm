from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import OpenAI

from ConfigReader import OPENAI_API_TOKEN


class LLM(object):
    def __init__(self):
        self.__OPEN_AI_KEY: str = OPENAI_API_TOKEN
        self.__chain: RunnableSerializable[dict, str]

    @property
    def __get_context(self):
        return """Instructions/Procedure:
            You will get a phrase or a question. You need to get main meaning.
            pick 1 from the following. If nothing is alligned. Pick closest one:
            1. user greeted
            2. user say good buy
            3. user asked what llm capabilities are
            4. user asked something meaningless
            5. user asked something about Cyprus (island)
            
            user's phrase: {question}
        """

    @property
    def __parse(self):
        return """
            Pick appropriate phrase with meaning: {context}. Return nothing but phrase. No dots, numbers, additional words.
                Hi
                Good buy
                Im a helpfull assistant
                Please, say it again
                Cyprus is a great island (if Cyprus is mentioned)
        """

    @property
    def __context_template(self) -> RunnableSerializable:
        return PromptTemplate.from_template(self.__get_context)

    @property
    def __main_template(self) -> RunnableSerializable:
        return PromptTemplate.from_template(self.__parse)

    def get_response(self, TEXT: str) -> str:
        llm: OpenAI = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            openai_api_key=self.__OPEN_AI_KEY,
            max_tokens=-1,
        )
        context_llm_chain: RunnableSerializable[dict, str] = self.__context_template | llm
        CONTEXT: str = context_llm_chain.invoke(input={
            "question": TEXT
        })
        main_llm_chain: RunnableSerializable[dict, str] = self.__main_template | llm
        print(f"context: {CONTEXT}")
        RESPONSE: str = main_llm_chain.invoke(input={
            "context": CONTEXT
        })
        print(f"response: {RESPONSE}")
        return RESPONSE
