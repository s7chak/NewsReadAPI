import json
import os
import time
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

import newsops
# nltk.download('stopwords')
# nltk.download('punkt')


app = Flask(__name__)
CORS(app)

today_date = datetime.today().strftime('%Y%m%d')
bucket_name = 'story-store'
func_start_time = time.time()
json_file_path = 'sources.json'
feed_types = ['blogml']
feedtitle = 'all' if feed_types=='all' else '-'.join(feed_types)
imgpath = 'img/'
store_mode = 'file'
relev_fields = ['OTitle', 'Link', 'Type', 'Date']
subset_data = None

@app.route('/start_fetch', methods=['GET'])
def story_fetch():
    print("Starting daily story fetch...")
    start_time = time.time()
    sources = load_sources()
    global feed_types, feedtitle, subset_data
    types = request.args.get('type')
    if types is not None and types != '':
        feed_types = types if 'all' in types else types.split(' ')
    if feed_types == 'all':
        feed_types = list(sources.keys())
        feedtitle = 'all'
    else:
        feedtitle = '-'.join([f.lower() for f in feed_types])

    data = get_data(sources)
    # res = newsops.study_stories(data)
    # print_result(res)

    end_time = time.time()
    elapsed_time = end_time - start_time
    analyze_elapsed_time = time.time() - end_time
    total_elapsed_time = time.time() - func_start_time
    print(f"Daily Story Fetch Time: {elapsed_time} seconds")
    print(f"Analyze Story Time: {analyze_elapsed_time} seconds")
    print(f"Run Time: {total_elapsed_time} seconds")
    subset = data[relev_fields].to_dict(orient='records')
    subset_data = data
    feeds = ', '.join(feed_types) if feed_types!='all' else ''
    total = data.shape[0]
    return make_response(jsonify({'data': subset, 'types': feeds, 'total': total}), 200)

@app.route('/search_stories', methods=['GET'])
def search_stories():
    print("Starting daily story fetch...")
    start_time = time.time()
    global feedtitle, feed_types, subset_data
    search_term = request.args.get('search_term')
    terms = search_term.split(' ')
    filename = today_date + '-' + feedtitle+'.csv'
    data = pd.read_csv(filename, index_col=None)
    result = newsops.search_articles(terms, data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Search Story Time: {elapsed_time} seconds")
    subset = result[relev_fields].to_dict(orient='records')
    subset_data = result
    feeds = ', '.join(feed_types) if feed_types != 'all' else ''
    count = result.shape[0]
    return make_response(jsonify({'data': subset, 'types': feeds, 'total': count }), 200)

@app.route('/word_clouds', methods=['GET'])
def wordcloud():
    global subset_data
    if subset_data is not None:
        fields = request.args.get('fields').split(',') if 'fields' in request.args else ['PTitle']
        res = newsops.do_wordclouds(subset_data, fields)
        response = make_response(jsonify({'wc': res['wc'], 'top20': res['top20']}), 200)
        return response

@app.route('/summary', methods=['GET'])
def summary():
    global subset_data
    if subset_data is not None:
        fields = request.args.get('fields').split(',') if 'fields' in request.args else ['PTitle']
        res = newsops.basic_summary(subset_data, fields)
        response = make_response(jsonify({'summary': res}), 200)
        return response


def get_data(sources):
    if store_mode == 'file':
        filename = today_date + '-' + feedtitle + '.csv'
        if not os.path.exists(filename):
            soups = newsops.fetch_article_soups(sources, feed_types)
            data = newsops.process_article_soups(soups)
            newsops.do_text_preprocessing(data)
            data.to_csv(filename, index=False)
        else:
            print("New articles not fetched, using saved data: " + today_date)
            data = pd.read_csv(filename, index_col=None)
        return data

def load_sources():
    with open(json_file_path, 'r') as json_file:
        sources = json.load(json_file)
    return sources
def print_result(res):
    global feed_types
    for i in res:
        if i == 'wc':
            for name, img in res[i].items():
                img_name = name+'-'+feedtitle
                with open(imgpath+img_name+'.png', 'wb') as file:
                    file.write(img)
        else:
            print(res[i])

if __name__ == '__main__':
    app.run(port=8091)

# if __name__ == "__main__":
#     start_time = time.time()
#     result = story_fetch()
#     filename = today_date + '-' + feedtitle+'.csv'
#     data = pd.read_csv(filename, index_col=None)
#     result = newsops.search_articles("gold", data)
#     print("Found: ", str(result.shape[0]))
#     print(result.head())
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Total Process Time: {elapsed_time} seconds")