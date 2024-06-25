from LLM import LLM
from flask import Flask, request


app = Flask(__name__)
llm_: LLM = LLM()


@app.route('/invoke/instagram/', methods=['GET'])
def instagram_llm():
    TASK: str = request.args.get('task')
    return llm_.get_response_instagram(TASK=TASK)

@app.route('/invoke/tokens/', methods=['GET'])
def tokens():
    TEXT: str = request.args.get("text")
    return llm_.get_tokens(TEXT=TEXT)

@app.route('/invoke/embeddings/', methods=['GET'])
def embeddings():
    TEXT: str = request.args.get("text")
    return llm_.get_embeddings(TEXT=TEXT)

@app.route('/invoke/warhammer40k/', methods=['GET'])
def warhammer40k_llm():
    TASK: str = request.args.get('task')
    CONTEXT: str = request.args.get('context')
    PAGE_NUM: str = request.args.get('page_num')
    return llm_.get_response_warhammer(TASK=TASK, CONTEXT=CONTEXT, PAGE_NUM=PAGE_NUM)


app.run(port=8082, debug=True)
