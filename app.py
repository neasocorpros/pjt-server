# app.py => Serving Logic
from flask import Flask, request

# pip install flask_cors
from flask_cors import CORS

from utils.llm import query_llm

app = Flask(__name__) # flask 서버 그 자체 / app이 아니라 다른 변수명도 노상관. 국룰일뿐
CORS(app)

# app.run() => 위 app을 돌리는 코드임 / 여기까지 세 줄이면 서버 돌기 시작

# not found 뜨는 이유. route가 없어서. 뒤에 아무것도 없어서. "GET / "
# 서버는 도는데, 오픈한 매장인데 메뉴가 없음. 줄수 있는 게 없음. 메뉴판 만들어야.

# Routing (특정 URL마다 어떤 응답을 줄지 정의)
@app.route('/', methods=['POST']) # 밑에 함수로 이때마다 어떤 일을 할지를 써놔야
def index():
    user_input = request.json['message']
    ans = query_llm(user_input)
    return {'llm': ans}

# '/hi' => 사용자에게 {message: 'hi'} 을 응답으로 제공
@app.route('/hi')
def hi():
    return {'message': 'hi'} #JS는 따옴표 안 써도 되지만 파이썬은 써야 함

app.run(debug=True)