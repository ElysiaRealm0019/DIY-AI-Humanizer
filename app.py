from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import requests
import json
import configparser
import os
import spacy
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize  # 确保在这里导入
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser  # 如果使用，保持引入
from collections import Counter
import random

app = Flask(__name__)

# 加载 spaCy 模型用于句子分割
nlp = spacy.load("en_core_web_md")

# 下载必要的 NLTK 资源
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
stop_words = set(stopwords.words('english'))

# --- 配置文件处理 ---
config = configparser.ConfigParser()
CONFIG_FILE = 'config.ini'

def create_default_config():
    """创建默认配置文件。"""
    config['API_KEYS'] = {
        'gemini': '',
        'siliconflow': '',
        'sapling': '',
        'winston': ''
    }
    config['MODELS'] = {
        'gemini': 'gemini-pro',
        'siliconflow': 'deepseek-ai/DeepSeek-R1'
    }
    config['SETTINGS'] = {
        'complexity_preference': 'high',
        'vocabulary_preference': 'formal',
        'paraphrase_strategy': 'conservative',
        'max_synonym_candidates': '3',
        'similarity_threshold': '0.7',
        'context_window': '3',
        'enable_sentence_variation': 'False',  # 新增：是否启用句子结构变化
    }
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def load_config():
    """加载配置文件，如果文件不存在则创建。"""
    if not os.path.exists(CONFIG_FILE):
        create_default_config()
    config.read(CONFIG_FILE)

load_config()

@app.route('/get_config', methods=['GET'])
def get_config_route():
    """
    返回当前配置信息。
    """
    load_config()  # 也要在这里重新加载！
    settings_dict = dict(config['SETTINGS'])
    return jsonify(settings_dict)

load_config()

# 模拟的 API 用量计数器
api_usage = {
    "gemini": 0,
    "siliconflow": 0,
    "sapling": 0,
    "winston": 0
}

reference_vocab = Counter()
is_initialized = False  # 标志位，确保只加载一次

def load_reference_text(file_path):
    """加载参考文章并统计词汇频率"""
    global reference_vocab
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # 使用更精细的句子和单词分词
        sentences = sent_tokenize(text)  # 添加这一行来定义 sentences
        words = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if word.isalpha()]
        filtered_words = [word for word in words if word not in stop_words]
        reference_vocab = Counter(filtered_words)
        print(f"Loaded reference text with {len(reference_vocab)} unique words.")
    else:
        print(f"Reference text file {file_path} not found.")


@app.before_request
def initialize_before_request():
    """在第一次请求前加载参考文章"""
    global is_initialized
    if not is_initialized:
        load_reference_text('reference_text.txt')  # 调整为您的实际文件路径
        is_initialized = True
        print("Initialized reference text.")

# 新增学术化同义词获取函数，参考文章词汇, 更精细的控制
def get_academic_synonyms(word, pos_tag, context_words, max_candidates=3, similarity_threshold=0.7):
    wordnet_pos = {'NN': 'n', 'VB': 'v', 'JJ': 'a', 'RB': 'r'}.get(pos_tag[:2], None)
    if not wordnet_pos or not context_words:
        return [word]

    synonyms = set()
    for syn in wordnet.synsets(word, pos=wordnet_pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if (synonym != word.lower() and synonym.isalpha() and
                len(synonym.split()) == 1 and synonym not in stop_words and len(synonym) <= 15):
                synonyms.add(synonym)
    
    context_doc = nlp(' '.join(context_words))
    if not context_doc.has_vector:
        return [word] if not synonyms else list(synonyms)[:max_candidates]

    prioritized_synonyms = []
    for synonym in synonyms:
        syn_doc = nlp(synonym)
        if syn_doc.has_vector:
            word_similarity = nlp(word).similarity(syn_doc)
            context_similarity = syn_doc.similarity(context_doc)
            combined_score = word_similarity * context_similarity
            if word_similarity >= similarity_threshold:
                prioritized_synonyms.append((synonym, combined_score))

    prioritized_synonyms.sort(key=lambda x: x[1], reverse=True)
    return [syn[0] for syn in prioritized_synonyms[:max_candidates]] or [word]


def adjust_morphology(original_word, synonym, original_tag):
    """
    调整同义词的形态以匹配原词的时态或单复数。
    使用 WordNetLemmatizer，并处理不规则变化。
    """
    lemmatizer = WordNetLemmatizer()

    if original_tag.startswith('VB'):  # 动词
        if original_tag == 'VBD':  # 过去式
            return lemmatizer.lemmatize(synonym, 'v') + 'ed'
        elif original_tag == 'VBG':  # 现在分词
            return lemmatizer.lemmatize(synonym, 'v') + 'ing'
        elif original_tag == 'VBN':  # 过去分词
            # 对于不规则动词，WordNetLemmatizer 可能无法正确处理，这里尝试做一些改进
            past_participle = lemmatizer.lemmatize(synonym, 'v')
            if past_participle == synonym:  # 如果没有变化，可能是原形，尝试加ed
                 past_participle = synonym + 'ed'
            return past_participle
        elif original_tag == 'VBZ': # 第三人称单数
            return lemmatizer.lemmatize(synonym, 'v') + 's'
        else: # 其他动词形式
            return lemmatizer.lemmatize(synonym, 'v')


    elif original_tag.startswith('NN'):  # 名词
        if original_tag.endswith('S'):  # 复数
            # 更可靠的复数形式处理 (但仍可能有少数不规则复数处理不好)
            return lemmatizer.lemmatize(synonym, 'n') + 's'
        else: # 单数
            return lemmatizer.lemmatize(synonym, 'n')
    return synonym

def vary_sentence_structure(doc):
    """
    通过调整句子结构来增加文本的多样性。
    """
    sentences = list(doc.sents)
    new_sentences = []
    complexity_preference = config.get('SETTINGS', 'complexity_preference', fallback='high')  # 从配置读取

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        if complexity_preference == 'high' and i + 1 < len(sentences):
            next_sent = sentences[i+1]
            if len(sent) < 20 and len(next_sent) < 20:
                combined = f"{sent.text.rstrip('.?!')} and {next_sent.text}"
                new_sentences.append(combined)
                i += 2
                continue
        elif complexity_preference == 'low' and len(sent) > 30:
            for token in sent:
                if token.dep_ in ('cc', 'relcl'):
                    first_part = sent[:token.i - sent.start].text
                    second_part = sent[token.i - sent.start:].text
                    new_sentences.extend([first_part, second_part])
                    i += 1
                    break  # 添加 break 以避免重复处理
            else:
                new_sentences.append(sent.text)
                i += 1
        new_sentences.append(sent.text)
        i += 1

    return " ".join(new_sentences)

def introduce_discourse_markers(text):
    """
    引入话语标记（discourse markers）来增强句子之间的连贯性。
    """
    # 更全面的话语标记列表，根据语义关系分类
    markers = {
        'addition': ['furthermore', 'moreover', 'in addition', 'besides', 'also', 'additionally'],
        'contrast': ['however', 'nevertheless', 'nonetheless', 'on the other hand', 'conversely', 'by contrast', 'yet'],
        'cause': ['because', 'since', 'as', 'due to', 'owing to'],
        'effect': ['therefore', 'thus', 'consequently', 'as a result', 'hence', 'accordingly'],
        'example': ['for example', 'for instance', 'such as', 'to illustrate', 'namely'],
        'emphasis': ['indeed', 'in fact', 'certainly', 'notably', 'especially'],
        'conclusion': ['in conclusion', 'to conclude', 'to summarize', 'in summary', 'finally'],
        'sequence': ['firstly', 'secondly', 'thirdly', 'next', 'then', 'finally', 'subsequently'],
        'clarification': ['in other words', 'that is to say', 'specifically', 'to clarify', 'put differently'],
    }
    sentences = sent_tokenize(text)
    new_sentences = []

    for i, sentence in enumerate(sentences):
        new_sentences.append(sentence)
        if i < len(sentences) - 1:
            # 基于与下一句的关系选择合适的话语标记 (需要更复杂的逻辑, 这里只是简单示例)
            next_sentence = sentences[i+1]

            # 示例：检测因果关系 (非常简化)
            if "because" in next_sentence.lower() or "since" in next_sentence.lower() or "due to" in next_sentence.lower():
                marker = random.choice(markers['cause'])
                new_sentences.append(f"{marker}, ")

            # 示例: 检测对比关系
            elif "but" in next_sentence.lower() or "however" in next_sentence.lower():
                 marker = random.choice(markers['contrast'])
                 new_sentences.append(f"{marker}, ")

            # 示例: 补充关系
            elif i > 0:
                marker = random.choice(markers['addition'])
                new_sentences.append(f"{marker}, ")
    return " ".join(new_sentences)



def paraphrase_text_local(sentence):
    """
    学术化改写句子，更注重保持原意和流畅度。
    """
    doc = nlp(sentence)
    words = [token.text for token in doc]
    pos_tags = nltk.pos_tag(words)
    new_tokens = []

    # 从配置文件读取参数
    paraphrase_strategy = config.get('SETTINGS', 'paraphrase_strategy', fallback='conservative')
    max_synonym_candidates = int(config.get('SETTINGS', 'max_synonym_candidates', fallback=3))
    similarity_threshold = float(config.get('SETTINGS', 'similarity_threshold', fallback=0.7))
    context_window = int(config.get('SETTINGS', 'context_window', fallback=3))
    enable_sentence_variation = config.getboolean('SETTINGS', 'enable_sentence_variation', fallback=False) # 读取启用标志
    lemmatizer = WordNetLemmatizer()
    # ... (同义词替换等逻辑，与之前版本相同) ...
    for i, (word, tag) in enumerate(pos_tags):
        token = doc[i]

        if tag.startswith(('NN', 'VB', 'JJ', 'RB')) and word.lower() not in stop_words:
            # --- 更严格的依存关系保护 ---
            if token.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'ROOT', 'pobj', 'csubj', 'csubjpass', 'expl'):  # 增加更多核心成分
                new_tokens.append(word)
                continue

            # 获取上下文单词
            start = max(0, i - context_window)
            end = min(len(words), i + context_window + 1)
            context_words = [words[j] for j in range(start, end) if j != i] # 排除自身

            synonyms = get_academic_synonyms(word, tag, context_words, max_synonym_candidates, similarity_threshold)

            if synonyms:
                # 策略：选择最佳同义词 (保守策略)
                synonym = synonyms[0]  #  选择评分最高的
                adjusted_word = adjust_morphology(word, synonym, tag)
                new_tokens.append(adjusted_word)
            else:
                new_tokens.append(word)
        else:
            new_tokens.append(word)

    # --- 基于规则的后处理 (更全面的语法检查) ---
    paraphrased_text = ' '.join(new_tokens)
    paraphrased_doc = nlp(paraphrased_text)
    final_sentence = []

    for i, token in enumerate(paraphrased_doc):
        # 1. 冠词检查 (更精细)
        if token.pos_ == 'NOUN' and token.dep_ != 'compound':
            if i > 0:
                prev_token = paraphrased_doc[i - 1]
                if prev_token.text.lower() not in ('a', 'an', 'the'):
                    if token.text[0].lower() in 'aeiou':
                        final_sentence.append('an')
                    else:
                        final_sentence.append('a')

        # 2. 动词一致性检查 (简单规则)
        elif token.tag_ == 'VBZ' and i > 0:
            prev_token = paraphrased_doc[i-1]
            if prev_token.tag_ not in ('NN', 'PRP', 'NNP') or prev_token.text.lower() == 'i':  # 简单的主谓一致
                # 尝试还原为原形
                lemma = lemmatizer.lemmatize(token.text, 'v')
                final_sentence.append(lemma)
                continue

        # 3. 介词搭配检查 (非常有限，需要更复杂的规则库或模型)
        elif token.pos_ == 'ADP':  # 介词
             if token.text.lower() == 'in' and i > 0:
                 prev_token = paraphrased_doc[i-1]
                 if prev_token.pos_ == "VERB" and prev_token.lemma_ not in ("be", "result", "consist", "participate", "believe", "succeed"):
                      # 如果前面的动词通常不与 "in" 搭配, 尝试替换为其他介词 (这里非常简化)
                      final_sentence.append("of") # 示例替换
                      continue
        final_sentence.append(token.text)

    final_text = ' '.join(final_sentence)


    # 引入话语标记
    final_text = introduce_discourse_markers(final_text)

    # 根据标志决定是否进行句子结构变化
    if enable_sentence_variation:
        final_text = vary_sentence_structure(nlp(final_text))

    return final_text[0].upper() + final_text[1:]


@app.route('/update_config', methods=['POST'])
def update_config_route():
    """
    接收前端发送的配置更新请求，并更新配置文件。
    """
    new_config = request.get_json()

    # 验证配置项的合法性
    valid_settings = {
        'complexity_preference': ['high', 'medium', 'low'],
        'vocabulary_preference': ['formal', 'semi-formal', 'informal'],
        'paraphrase_strategy': ['conservative', 'balanced', 'aggressive'],
        'max_synonym_candidates': lambda x: x.isdigit() and 1 <= int(x) <= 10,
        'similarity_threshold': lambda x: x.replace('.', '', 1).isdigit() and 0 <= float(x) <= 1,
        'context_window': lambda x: x.isdigit() and 1 <= int(x) <= 10,
        'enable_sentence_variation': lambda x: str(x).lower() in ('true', 'false')  # 修改为 lambda 函数，接受大小写
    }

    for key, value in new_config.items():
        if key not in valid_settings:
            return jsonify({'error': f'Invalid config key: {key}'}), 400
        if isinstance(valid_settings[key], list) and value not in valid_settings[key]:
            return jsonify({'error': f'Invalid value for {key}: {value}'}), 400
        if callable(valid_settings[key]) and not valid_settings[key](value):
            return jsonify({'error': f'Invalid value for {key}: {value}'}), 400

        # 规范化 enable_sentence_variation 的值
        if key == 'enable_sentence_variation':
            value = 'True' if str(value).lower() == 'true' else 'False'  # 转换为大写形式
        config['SETTINGS'][key] = value

    # 更新 config.ini 文件
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    return jsonify({'message': 'Config updated successfully'})

# API 调用函数
def call_deepseek_api(api_key, model, prompt, text):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        full_response = data['choices'][0]['message']['content']
        print("DeepSeek Output:", full_response)
        api_usage['siliconflow'] += 1
        return full_response, None
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API 请求错误: {e}")
        return None, str(e)
    except KeyError as e:
        print(f"DeepSeek API 响应缺少键: {e}")
        return None, f"DeepSeek API 响应格式错误，缺少键: {e}"
    except Exception as e:
        print(f"DeepSeek API 处理错误: {e}")
        return None, str(e)

def call_gemini_api(api_key, model, prompt, text):
    try:
        genai.configure(api_key=api_key)
        if not model:
            return None, "未选择Gemini模型"
        gemini_model = genai.GenerativeModel(model_name=model)
        response = gemini_model.generate_content(prompt + "\n" + text, stream=True)
        full_response = ""
        for chunk in response:
            full_response += chunk.text
        api_usage['gemini'] += 1
        return full_response, None
    except Exception as e:
        print(f"Gemini API 请求错误: {e}")
        return None, str(e)

def call_sapling_api(api_key, text):
    try:
        response = requests.post(
            "https://api.sapling.ai/api/v1/aidetect",
            json={"key": api_key, "text": text}
        )
        response.raise_for_status()
        result = response.json()
        api_usage['sapling'] += 1
        return result
    except requests.exceptions.RequestException as e:
        print(f"Sapling API 请求错误: {e}")
        return {"score": "N/A", "sentence_scores": []}
    except Exception as e:
        print(f"Sapling API 处理错误: {e}")
        return {"score": "N/A", "sentence_scores": []}

def call_winston_api(api_key, text):
    try:
        response = requests.post(
            "https://api.gowinston.ai/v2/ai-content-detection",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"text": text, "sentences": True}
        )
        response.raise_for_status()
        result = response.json()
        print("Winston API Response:", json.dumps(result, indent=2))
        api_usage['winston'] += 1
        return result
    except requests.exceptions.RequestException as e:
        print(f"Winston API 请求错误: {e}")
        return {"status": "N/A", "score": -1, "sentences": [], "credits_used": "N/A", "credits_remaining": "N/A", "readability_score": "N/A"}
    except Exception as e:
        print(f"Winston API 处理错误: {e}")
        return {"status": "N/A", "score": -1, "sentences": [], "credits_used": "N/A", "credits_remaining": "N/A", "readability_score": "N/A"}

def rewrite_sentence(selected_model, gemini_api_key, gemini_model, siliconflow_api_key, siliconflow_model, prompt, sentence):
    """根据用户选择的模型重写句子，API 失败时回退到本地改写"""
    if selected_model == "gemini" and gemini_api_key:
        rewrite_result, error = call_gemini_api(gemini_api_key, gemini_model, prompt, sentence)
        if rewrite_result:
            return rewrite_result, None
        print(f"Gemini failed, falling back to local: {error}")
    elif selected_model == "siliconflow" and siliconflow_api_key:
        rewrite_result, error = call_deepseek_api(siliconflow_api_key, siliconflow_model, prompt, sentence)
        if rewrite_result:
            return rewrite_result, None
        print(f"SiliconFlow failed, falling back to local: {error}")

    print(f"Using local academic paraphrasing for model: {selected_model}")
    return paraphrase_text_local(sentence), None

@app.route('/')
def index():
    return render_template('index.html', config=config)

@app.route('/get_gemini_models')
def get_gemini_models():
    api_key = request.args.get('gemini_api_key')
    if not api_key:
        return jsonify({'error': 'Gemini API 密钥缺失'}), 400
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败：{str(e)}'}), 500

@app.route('/get_siliconflow_models')
def get_siliconflow_models():
    api_key = request.args.get('siliconflow_api_key')
    if not api_key:
        return jsonify({'error': 'SiliconFlow API 密钥缺失'}), 400
    url = "https://api.siliconflow.cn/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        models = [item['id'] for item in data['data']]
        return jsonify({'models': models})
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'获取 SiliconFlow 模型列表失败: {e}'}), 500
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'解析 SiliconFlow 模型列表响应失败: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'获取 SiliconFlow 模型列表时发生未知错误: {e}'}), 500

def get_siliconflow_balance(api_key):
    """获取 SiliconFlow 账户余额。"""
    url = "https://api.siliconflow.cn/v1/user/info"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data['data']['totalBalance']
    except requests.exceptions.RequestException as e:
        print(f"获取 SiliconFlow 余额失败: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"解析 SiliconFlow 余额响应失败: {e}")
        return None
    except Exception as e:
        print(f"获取 SiliconFlow 余额时发生未知错误: {e}")
        return None

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    input_text = data['input_text']
    prompt = data['prompt']
    selected_model = data['selected_model']  # "gemini", "siliconflow", 或 "local"
    gemini_api_key = data.get('gemini_api_key', config.get('API_KEYS', 'gemini', fallback=''))
    siliconflow_api_key = data.get('siliconflow_api_key', config.get('API_KEYS', 'siliconflow', fallback=''))
    sapling_api_key = data.get('sapling_api_key', config.get('API_KEYS', 'sapling', fallback=''))
    winston_api_key = data.get('winston_api_key', config.get('API_KEYS', 'winston', fallback=''))
    detector = data['detector']
    gemini_model = data.get('gemini_model', config.get('MODELS', 'gemini', fallback='gemini-pro'))
    siliconflow_model = data.get('siliconflow_model', config.get('MODELS', 'siliconflow', fallback='deepseek-ai/DeepSeek-R1'))
    sentence_threshold = float(data.get('sentence_threshold', 0.8))
    paragraph_threshold = float(data.get('paragraph_threshold', 0.1))

    # 更新配置文件（仅当选择对应模型时）
    if selected_model == "gemini" and gemini_api_key:
        config['API_KEYS']['gemini'] = gemini_api_key
        if gemini_model:
            config['MODELS']['gemini'] = gemini_model
    elif selected_model == "siliconflow" and siliconflow_api_key:
        config['API_KEYS']['siliconflow'] = siliconflow_api_key
        if siliconflow_model:
            config['MODELS']['siliconflow'] = siliconflow_model
    if sapling_api_key:
        config['API_KEYS']['sapling'] = sapling_api_key
    if winston_api_key:
        config['API_KEYS']['winston'] = winston_api_key
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    # 初始文本和迭代计数
    current_text = input_text
    iteration = 0
    results = []
    max_iterations = 10
    previous_score = None

    print(f"Starting with sentence_threshold={sentence_threshold}, paragraph_threshold={paragraph_threshold}, selected_model={selected_model}")

    while iteration < max_iterations:
        print(f"Iteration {iteration + 1}: Current Text: {current_text}")

        # 1. 获取句子级别的 AI 得分
        if detector == 'sapling':
            paragraph_response = call_sapling_api(sapling_api_key, current_text)
            if isinstance(paragraph_response.get("score"), str):
                return jsonify({'error': 'Sapling API 调用失败', 'details': paragraph_response}), 500
            sentence_scores = paragraph_response.get('sentence_scores', [])
            current_score = paragraph_response['score']
            print(f"Sapling - Paragraph Score: {current_score}, Sentence Scores: {sentence_scores}")
        else:  # detector == 'winston'
            paragraph_response = call_winston_api(winston_api_key, current_text)
            if paragraph_response.get("score") == -1:
                return jsonify({'error': 'Winston API 调用失败', 'details': paragraph_response}), 500
            sentence_scores = [{"sentence": s["text"], "score": s["score"] / 100} for s in paragraph_response.get('sentences', [])]
            current_score = paragraph_response['score'] / 100
            print(f"Winston - Paragraph Score: {current_score}, Sentence Scores: {sentence_scores}")

        # 2. 直接使用 API 返回的句子
        sentences = [s['sentence'] for s in sentence_scores]
        print(f"Sentences from API: {sentences}")
        sentences_to_rewrite = []
        new_sentences = sentences.copy()

        # 3. 找出需要重写的句子
        for i, sentence in enumerate(sentences):
            score = sentence_scores[i]['score']
            if detector == 'sapling':
                if score > max(sentence_threshold, 0.5):
                    sentences_to_rewrite.append((i, sentence))
                    print(f"Sapling - Sentence {i}: '{sentence}' (Score: {score}) needs rewrite")
            else:
                if score < sentence_threshold:
                    sentences_to_rewrite.append((i, sentence))
                    print(f"Winston - Sentence {i}: '{sentence}' (Score: {score}) needs rewrite")

        # 4. 如果没有句子需要重写，但段落得分未达标
        if not sentences_to_rewrite:
            if detector == 'sapling' and current_score > paragraph_threshold:
                max_score_idx = max(range(len(sentence_scores)), key=lambda i: sentence_scores[i]['score'])
                sentences_to_rewrite.append((max_score_idx, sentences[max_score_idx]))
                print(f"Sapling - No sentences above threshold, rewriting highest scoring sentence {max_score_idx}: '{sentences[max_score_idx]}' (Score: {sentence_scores[max_score_idx]['score']})")
            elif detector == 'winston' and current_score < paragraph_threshold:
                min_score_idx = min(range(len(sentence_scores)), key=lambda i: sentence_scores[i]['score'])
                sentences_to_rewrite.append((min_score_idx, sentences[min_score_idx]))
                print(f"Winston - No sentences below threshold, rewriting lowest scoring sentence {min_score_idx}: '{sentences[min_score_idx]}' (Score: {sentence_scores[min_score_idx]['score']})")
            else:
                print("Paragraph score meets threshold, exiting loop")
                break

        # 5. 重写所有需要重写的句子
        all_failed = True
        for index, sentence in sentences_to_rewrite:
            rewrite_result, error = rewrite_sentence(selected_model, gemini_api_key, gemini_model, siliconflow_api_key, siliconflow_model, prompt, sentence)
            if rewrite_result:
                new_sentences[index] = rewrite_result
                results.append({
                    "model": selected_model if selected_model in ["gemini", "siliconflow"] and not error else "local",
                    "original_sentence": sentence,
                    "rewritten_sentence": rewrite_result,
                    "sapling_score": sentence_scores[index]['score'] if detector == 'sapling' else "N/A",
                    "winston_score": sentence_scores[index]['score'] * 100 if detector == 'winston' else "N/A"
                })
                print(f"Rewrote sentence {index}: '{sentence}' -> '{rewrite_result}'")
                all_failed = False
            else:
                new_sentences[index] = sentence
                results.append({
                    "model": "重写错误",
                    "original_sentence": sentence,
                    "rewritten_sentence": f"重写句子 '{sentence}' 时出错: {error}",
                    "sapling_score": sentence_scores[index]['score'] if detector == 'sapling' else "N/A",
                    "winston_score": sentence_scores[index]['score'] * 100 if detector == 'winston' else "N/A"
                })
                print(f"Error rewriting sentence {index}: {error}")

        # 如果所有重写都失败且未选择本地模型，提前退出
        if all_failed and selected_model != "local":
            print("All rewrite attempts failed, exiting loop")
            break

        # 6. 组合成新的段落
        current_text = ' '.join(new_sentences)
        print(f"New Text after rewrite: {current_text}")

        # 7. 检测整个段落的 AI 得分
        if detector == 'sapling':
            paragraph_response = call_sapling_api(sapling_api_key, current_text)
            if isinstance(paragraph_response.get("score"), str):
                return jsonify({'error': 'Sapling API 调用失败', 'details': paragraph_response}), 500
            current_score = paragraph_response['score']
            print(f"Sapling - New Paragraph Score: {current_score}")
        else:  # detector == 'winston'
            paragraph_response = call_winston_api(winston_api_key, current_text)
            if paragraph_response.get("score") == -1:
                return jsonify({'error': 'Winston API 调用失败', 'details': paragraph_response}), 500
            current_score = paragraph_response['score'] / 100
            print(f"Winston - New Paragraph Score: {current_score}")

        # 8. 检查重写效果
        if previous_score is not None and abs(previous_score - current_score) < 0.001:
            print(f"Warning: Minimal change in score ({previous_score} -> {current_score}), heavy rewriting may be ineffective")
            break
        previous_score = current_score

        # 9. 判断是否退出循环
        if detector == 'sapling':
            if current_score <= paragraph_threshold:
                print("Sapling score meets threshold, exiting loop")
                break
        else:
            if current_score >= paragraph_threshold:
                print("Winston score meets threshold, exiting loop")
                break

        iteration += 1

    # 如果达到最大迭代次数或提前退出，记录最后一次结果
    final_result = {
        "model": "最终结果（达到迭代上限）" if iteration >= max_iterations else "最终结果",
        "output": current_text,
    }
    if detector == 'sapling':
        final_result["sapling_score"] = current_score
        print(f"Final Sapling Score: {current_score}")
    else:  # winston
        final_result["winston_score"] = paragraph_response.get("score")
        final_result["credits_used"] = paragraph_response.get("credits_used", "N/A")
        final_result["readability_score"] = paragraph_response.get("readability_score", "N/A")
        print(f"Final Winston Score: {paragraph_response.get('score')}")

    results.append(final_result)

    # 准备返回数据，包括配额信息
    response_data = {
        "results": results,
        "gemini_usage": api_usage['gemini'] if selected_model == "gemini" else 0,
        "siliconflow_usage": api_usage['siliconflow'] if selected_model == "siliconflow" else 0,
        "sapling_usage": api_usage.get('sapling', 0),
        "winston_usage": api_usage.get('winston', 0),
    }
    if detector == 'winston':
        response_data["winston_balance"] = paragraph_response.get("credits_remaining", "N/A")
    if selected_model == "siliconflow" and siliconflow_api_key:
        response_data["siliconflow_balance"] = get_siliconflow_balance(siliconflow_api_key) or "N/A"

    return jsonify(response_data)


# 运行 Flask 应用
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')