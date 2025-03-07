from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import requests
import json
import configparser
import os
import spacy  # 引入 spaCy 进行句子分割

app = Flask(__name__)

# 加载 spaCy 模型用于句子分割
nlp = spacy.load("en_core_web_sm")

# --- 配置文件处理 ---
config = configparser.ConfigParser()
CONFIG_FILE = 'config.ini'

def load_config():
    """加载配置文件，如果文件不存在则创建。"""
    if not os.path.exists(CONFIG_FILE):
        create_default_config()
    config.read(CONFIG_FILE)

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
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

load_config()

# 模拟的 API 用量计数器
api_usage = {
    "gemini": 0,
    "siliconflow": 0,
    "sapling": 0,
    "winston": 0
}

# DeepSeek API 调用函数
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

# Gemini API 调用函数
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
        return full_response, None
    except Exception as e:
        print(f"Gemini API 请求错误: {e}")
        return None, str(e)

# Sapling API 调用函数
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

# Winston AI API 调用函数
def call_winston_api(api_key, text):
    try:
        response = requests.post(
            "https://api.gowinston.ai/v2/ai-content-detection",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"text": text, "sentences": True}
        )
        response.raise_for_status()
        result = response.json()
        print("Winston API Response:", json.dumps(result, indent=2))  # 调试输出
        api_usage['winston'] += 1
        return result
    except requests.exceptions.RequestException as e:
        print(f"Winston API 请求错误: {e}")
        return {"status": "N/A", "score": -1, "sentences": [], "credits_used": "N/A", "credits_remaining": "N/A", "readability_score": "N/A"}
    except Exception as e:
        print(f"Winston API 处理错误: {e}")
        return {"status": "N/A", "score": -1, "sentences": [], "credits_used": "N/A", "credits_remaining": "N/A", "readability_score": "N/A"}

def rewrite_sentence(gemini_api_key, gemini_model, siliconflow_api_key, siliconflow_model, prompt, sentence):
    """重写句子的函数，先尝试 Gemini，失败则尝试 DeepSeek。"""
    rewrite_result, error = call_deepseek_api(siliconflow_api_key, siliconflow_model, prompt, sentence)
    if rewrite_result:
        api_usage['siliconflow'] += 1
        return rewrite_result, None
    rewrite_result, error = call_gemini_api(gemini_api_key, gemini_model, prompt, sentence)
    if rewrite_result:
        api_usage['gemini'] += 1
        return rewrite_result, None
    return None, error

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
    gemini_api_key = data.get('gemini_api_key')
    siliconflow_api_key = data.get('siliconflow_api_key')
    sapling_api_key = data.get('sapling_api_key')
    winston_api_key = data.get('winston_api_key')
    detector = data['detector']
    gemini_model = data['gemini_model']
    siliconflow_model = data['siliconflow_model']
    sentence_threshold = float(data['sentence_threshold'])
    paragraph_threshold = float(data['paragraph_threshold'])

    # 更新配置文件
    if gemini_api_key:
        config['API_KEYS']['gemini'] = gemini_api_key
    if siliconflow_api_key:
        config['API_KEYS']['siliconflow'] = siliconflow_api_key
    if sapling_api_key:
        config['API_KEYS']['sapling'] = sapling_api_key
    if winston_api_key:
        config['API_KEYS']['winston'] = winston_api_key
    if gemini_model:
        config["MODELS"]["gemini"] = gemini_model
    if siliconflow_model:
        config["MODELS"]["siliconflow"] = siliconflow_model
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    # 初始文本和迭代计数
    current_text = input_text
    iteration = 0
    results = []
    max_iterations = 10
    previous_score = None

    print(f"Starting with sentence_threshold={sentence_threshold}, paragraph_threshold={paragraph_threshold}")

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

        # 3. 找出需要重写的句子（得分 > 0.5 以增加重写范围）
        for i, sentence in enumerate(sentences):
            score = sentence_scores[i]['score']
            if detector == 'sapling':
                if score > max(sentence_threshold, 0.5):  # 重写得分 > 0.5 的句子
                    sentences_to_rewrite.append((i, sentence))
                    print(f"Sapling - Sentence {i}: '{sentence}' (Score: {score}) needs rewrite")
            else:
                if score < sentence_threshold:
                    sentences_to_rewrite.append((i, sentence))
                    print(f"Winston - Sentence {i}: '{sentence}' (Score: {score}) needs rewrite")

        # 4. 如果没有句子需要重写，但段落得分未达标，重写得分最高的句子
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
            rewrite_result, error = rewrite_sentence(gemini_api_key, gemini_model, siliconflow_api_key, siliconflow_model, prompt, sentence)
            if rewrite_result:
                new_sentences[index] = rewrite_result
                results.append({
                    "model": "重写模型",
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

        # 如果所有重写都失败，提前退出
        if all_failed:
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
        if previous_score is not None and abs(previous_score - current_score) < 0.001:  # 放宽至 0.001
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
    if iteration >= max_iterations:
        print("Reached maximum iterations, exiting loop with final result")
        final_result = {
            "model": "最终结果（达到迭代上限）",
            "output": current_text,
        }
    else:
        final_result = {
            "model": "最终结果",
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

    # 准备返回数据
    response_data = {
        "results": results,
        "gemini_usage": api_usage['gemini'],
        "siliconflow_usage": api_usage['siliconflow'],
        "sapling_usage": api_usage.get('sapling', 0),
        "winston_usage": api_usage.get('winston', 0),
    }
    if detector == 'winston':
        response_data["winston_balance"] = paragraph_response.get("credits_remaining", "N/A")

    return jsonify(response_data)

# 运行 Flask 应用
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')