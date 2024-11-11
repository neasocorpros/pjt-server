# app.py => Serving Logic
from flask import Flask, request
from flask_cors import CORS

from utils.llm import query_llm

app = Flask(__name__) # flask 서버 그 자체
CORS(app)

# app.run() => 위 app을 돌리는 코드임 / 여기까지 세 줄이면 서버 돌기 시작

# Routing (특정 URL마다 어떤 응답을 줄지 정의)
@app.route('/', methods=['POST']) # 밑에 함수로 이때마다 어떤 일을 할지를 써놔야
def index():
    user_input = request.json['message']
    ans = query_llm(user_input)
    print("LLM response:", ans)  # 실제로 어떤 데이터가 오는지 확인

    movie_ids = []
    for doc in ans['generation']['documents']:
        print(doc, doc.metadata)
        mid = int(doc.metadata.get('id', -1))
        movie_ids.append(mid)
    
    llm = ans['generation']
    llm.pop('documents')
    response = {
        'llm': llm,
        'movieIds': movie_ids
        }
    print("Sending to client:", response)  # 클라이언트로 보내는 데이터 확인
    
    return response

# '/hi' => 사용자에게 {message: 'hi'} 을 응답으로 제공
@app.route('/hi')
def hi():
    return {'message': 'hello'} #JS는 따옴표 안 써도 되지만 파이썬은 써야 함

app.run(debug=True)
