from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from flask import Flask, render_template, request, json
from textblob import TextBlob
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

LANGUAGE = "english"


@app.route('/summary', methods=['POST'])
def json_example():
    req_data = request.get_json()
    full_text = req_data['full_text']
    num_of_sentences = req_data['num_of_sentences']
    SENTENCES_COUNT = num_of_sentences
    sentiment_value = TextBlob(full_text).sentiment.polarity

    parser = PlaintextParser.from_string(full_text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary = ''
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary += str(sentence) + '\n \n'

    return app.response_class(
        response=json.dumps({
            "sentiment_value": sentiment_value,
            "summary": summary
        }),
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run()