from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


# JSON 파일 경로 설정
skyline_json_path = './skylinedata/reviews_s.json'
topsis_json_path='./skylinedata/reviews_t.json'


# 엔드포인트 정의
@app.route('/skyline/result', methods=['GET'])
def get_skyline():
    try:
        with open(skyline_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

    
@app.route('/topsis/result', methods=['GET'])
def get_topsis():
    try:
        with open(topsis_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

# # JSON 파일에서 그룹화된 리뷰 데이터 로드
# with open('sorted_grouped_reviews.json', 'r', encoding='utf-8') as file:
#     grouped_reviews = json.load(file)

# @app.route('/get_reviews', methods=['GET'])
# def get_reviews():
#     return jsonify(grouped_reviews)

# @app.route('/get_review/<course_id>', methods=['GET'])
# def get_review_by_course_id(course_id):
#     # course_id에 해당하는 리뷰 데이터 찾기
#     if str(course_id) in grouped_reviews:
#         return jsonify({str(course_id): grouped_reviews[str(course_id)]})
#     else:
#         return jsonify({'error': 'Course_id not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)