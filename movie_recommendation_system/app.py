from flask import Flask, request, jsonify, render_template, json
from recommend_api import recommend_movie, movie_description, movie_title
from model import recommendation

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('index.html')

@app.route('/get_movies', methods=['GET'])
def return_movies():
    return jsonify({
        'movie_title' : movie_title,
        'movie_description' : movie_description
    })

@app.route('/recommend')
def recommend_movies():
    data = recommendation('castle and castle')
    return render_template('recommend_page.html', data=data)

@app.route('/castle')
def recommend_castle():
    data = recommendation('castle and castle')
    data = data.replace("{", "")
    data = data.replace("}", "")
    new_data = data.split(',')
    result_data = [i for i in new_data]
    return render_template('recommend_castle.html', data=result_data)

@app.route('/king_of_boys')
def recommend_kingOfBoys():
    data = recommendation('king of boys')
    return render_template('recommend_king_of_boys.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

