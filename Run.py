from LLM import LLM
from flask import Flask, request


app = Flask(__name__)
llm_: LLM = LLM()


@app.route('/invoke', methods=['GET'])
def llm():
    TEXT: str = request.args.get('text')
    RESPONSE: str = llm_.get_response(TEXT=TEXT)
    return RESPONSE


app.run(port=8082, debug=True)
