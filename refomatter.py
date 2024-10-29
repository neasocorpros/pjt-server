import json
from pprint import pprint



with open('database.jsonl', 'r') as f:
    movies = list(f)


with open('reformatted-db.jsonl', 'w', encoding='utf-8') as f:
    for movie_string in movies:
        movie = json.loads(movie_string)
        columns = [
            # 'id',
            'eventYear',
            'event',
            'historyDescription',
            'title',
            'genre',
            'movieDescription',
            # 'releaseDate',
            # 'rating',
            # 'numberOfMoviegoers',
            # 'runtime',
            'synopsys'
        ]
        
        movie['releaseDate'] = movie['releaseDate'].strip()
        
        oneline = ''
        for col in columns:
            oneline += col + ': ' + movie[col].strip() + '\n'
            movie.pop(col)    
        
        movie['summary'] = oneline
        f.write(json.dumps(movie) + '\n')