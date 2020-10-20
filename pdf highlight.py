# -*- coding: utf-8 -*-
"""
@Author: Fibonacci
@Time  : 2020/7/16 10:34
@Desc  : pdf自动高亮
"""
import re
import subprocess
import uuid
from collections import defaultdict, OrderedDict
import jieba
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from gensim.models import word2vec
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser, PDFDocument
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename


# 解析pdf文件为文本文件
def parse_pdf(input_file, output_file):
    with open(input_file, 'rb') as input_f:
        # 用文件对象创建一个PDF文档分析器
        parser = PDFParser(input_f)
        # 创建一个PDF文档
        doc = PDFDocument()
        # 分析器和文档相互连接
        parser.set_document(doc)
        doc.set_parser(parser)
        # 提供初始化密码，没有默认为空
        doc.initialize()
        # 检查文档是否可以转成TXT，如果不可以就忽略
        if not doc.is_extractable:
            raise PDFTextExtractionNotAllowed
        else:
            # 创建PDF资源管理器，来管理共享资源
            rsrcmagr = PDFResourceManager()
            # 创建一个PDF设备对象
            laparams = LAParams()
            # 将资源管理器和设备对象聚合
            device = PDFPageAggregator(rsrcmagr, laparams=laparams)
            # 创建一个PDF解释器对象
            interpreter = PDFPageInterpreter(rsrcmagr, device)

            content = ''
            # 循环遍历列表，每次处理一个page内容
            # doc.get_pages()获取page列表
            for page in doc.get_pages():
                interpreter.process_page(page)
                # 接收该页面的LTPage对象
                layout = device.get_result()
                # 这里的layout是一个LTPage对象 里面存放着page解析出来的各种对象
                # 一般包括LTTextBox，LTFigure，LTImage，LTTextBoxHorizontal等等一些对像
                # 想要获取文本就得获取对象的text属性
                with open(output_file, 'a', encoding='utf-8') as output_f:
                    for x in layout:
                        if isinstance(x, LTTextBoxHorizontal):
                            result = x.get_text()
                            content += result[:-1]
                            output_f.write(result)
    return content


# 获取停止词
def get_stop_words(file_name):
    stop_words = []
    with open(file_name, encoding='utf-8') as f:
        for word in f.readlines():
            stop_words.append(word.strip())
    return stop_words


# 获取文本分词结果
def get_words(content, stop_words):
    content_clean = ''.join(re.findall(r'[\d|\w]+', content))
    words = [w for w in jieba.cut(content_clean) if w not in stop_words and len(w) > 1]
    return words


# 获取词的联系
def get_word_relation(words, window_size=5, strides=1):
    words_connection = defaultdict(set)
    word_count = len(words)
    for i in range(0, word_count, strides):
        connected = words[i:i+window_size]
        for word in connected:
            words_connection[word].update(connected)
    for word in words_connection:
        words_connection[word].discard(word)
    return OrderedDict(words_connection)


# 初始化词的权重
def init_words_weight(words_connection):
    # 初始化节点权重
    words_weight = OrderedDict()
    for word in words_connection:
        words_weight[word] = 1 / len(words_connection)
    return words_weight


# 迭代更新词的权重至收敛
def update_weight(words_connection, words_weight, model, d=0.85, error_threshold=1e-5):
    def get_word_sum_weight(words_connection, words_weight, model, word, alpha=0.8):
        connected_words = words_connection[word]
        sum_weight = 0.0
        for connected_word in connected_words:
            relation_rate = max(0, 1 - cosine(model.wv[connected_word], model.wv[
                word])) if word in model.wv.vocab and connected_word in model.wv.vocab else 0
            sum_weight += (1 - alpha) * words_weight[connected_word] / len(
                words_connection[connected_word]) + alpha * relation_rate
        return sum_weight

    pre_weight = OrderedDict(words_weight)
    cur_weight = OrderedDict()
    while True:
        for word in pre_weight:
            cur_weight[word] = (1-d) + d * get_word_sum_weight(words_connection, pre_weight, model, word)
        errors = [abs(i-j) for i, j in zip(pre_weight.values(), cur_weight.values())]
        if all(np.array(errors) < error_threshold):
            return cur_weight
        else:
            pre_weight = cur_weight.copy()


# 获取关键词
def get_key_words(words_connection, model, top_k=5, similarity=0.75):
    words_weight = init_words_weight(words_connection)
    words_weight = update_weight(words_connection, words_weight, model)
    key_words = set(words_weight[0] for words_weight in sorted(words_weight.items(), key=lambda item: item[1], reverse=True)[:top_k])
    key_words_similar = []
    for word in words_weight:
        if any(cosine_similarity(model.wv[key_words], model.wv[word[0]].reshape(1, -1)) > similarity):
            key_words_similar.append(word[0])
    key_words.update(key_words_similar)
    return key_words


# 文本高亮
def highlight(input_file, output_file, key_words):
    with open(input_file, 'r', encoding='utf8') as f_r:
        html_content = f_r.read()
        for word in key_words:
            html_content = html_content.replace(word, '<span style="background-color:yellow;">' + word + '</span>')
        with open(output_file, 'w', encoding='utf8') as f_h:
            f_h.write(html_content)


model = word2vec.Word2Vec.load('D:/downloads/AI Courses/nlp核心课程/lesson09-assignment/wiki_corpus.model')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', file_name='', display='none')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        upload_file_name = secure_filename(file.filename)
        if upload_file_name == 'pdf':
            upload_file_name = str(uuid.uuid4()) + '.pdf'
        pdf_file = 'static/pdf/' + upload_file_name
        file.save(pdf_file)
        output_path = 'static/pdf'
        code = subprocess.call('D:/Program/pdf2htmlEX-win32-0.14.6/pdf2htmlEX.exe ' + pdf_file + ' --dest-dir ' + output_path)
        if code == 0:
            html_file = output_path + '/' + upload_file_name[:-3] + 'html'
            content = parse_pdf(pdf_file, output_path + '/' + upload_file_name[:-3] + 'txt')
            stop_words = get_stop_words('D:/downloads/AI Courses/nlp核心课程/datasource/百度停用词表.txt')
            words = get_words(content, stop_words)
            words_connection = get_word_relation(words, window_size=5, strides=3)
            key_words = get_key_words(words_connection, model, top_k=7)
            highlight_file = output_path + '/' + upload_file_name[:-4] + '_highlight.html'
            highlight(html_file, highlight_file, key_words)
            return render_template('index.html', file_name=highlight_file, display='inline')
        else:
            return 'Server Error'
    else:
        return render_template('index.html')


if __name__ == '__main__':
    # 设置调试模式，生产模式的时候要关掉debug
    # app.debug = True
    app.run()

