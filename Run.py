from LLM import LLM
from flask import Flask, request


app = Flask(__name__)
llm_: LLM = LLM()


@app.route('/invoke', methods=['GET'])
def llm():
    TASK: str = request.args.get('task')
    RESPONSE: str = llm_.get_response(TASK=TASK)
    return RESPONSE


app.run(port=8082, debug=True)
