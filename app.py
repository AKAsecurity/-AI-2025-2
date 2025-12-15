# app.py
import os
import requests
import xml.etree.ElementTree as ET
import datetime as dt
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Google GenAI SDK
from google import genai
from google.genai import types

# .envファイルから環境変数を読み込む
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- Gemini API functions ---

def chat(request_prompt):
    """
    Gemini APIを呼び出し、ニュースのタグを生成する。
    """
    if not GEMINI_API_KEY:
        print("エラー: GEMINI_API_KEYが設定されていません。")
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)
    content_string = request_prompt['messages'][0]['content']
    
    # 応答生成の設定
    config = types.GenerateContentConfig(
        system_instruction=request_prompt['context'],
        max_output_tokens=request_prompt['maxOutputTokens'],
        temperature=request_prompt['temperature'],
        top_p=request_prompt['topP'],
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content_string,
            config=config,
        )
        
        # タグの整形: 応答をカンマなどで分割してリストにする
        # 例: "テクノロジー, AI, 新製品" -> ["テクノロジー", "AI", "新製品"]
        tags = [t.strip() for t in response.text.split(',') if t.strip()]
        return tags
        
    except Exception as e:
        print(f"Gemini API呼び出し中にエラーが発生しました: {e}")
        return [] # エラー時は空のリストを返す

def generate_request_prompt(prompt, content, tmp, p):
    """
    Gemini APIに送信するリクエストプロンプトを作成する。
    """
    request_prompt = {
        'context': prompt,
        'maxOutputTokens': 1024,
        'messages': [
            {
                'author': 'user',
                'content': content,
            }
        ],
        'temperature': tmp,
        'topP': p,
    }
    return request_prompt

# --- RSS Feed URLs (STEP1/2) ---
# 3つ以上のカテゴリと、対応するRSSフィードURLを設定
RSS_FEEDS = {
    'テクノロジー': 'https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=ja&gl=JP&ceid=JP:ja',
    '経済': 'https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=ja&gl=JP&ceid=JP:ja',
    'スポーツ': 'https://news.google.com/rss/headlines/section/topic/SPORTS?hl=ja&gl=JP&ceid=JP:ja',
    '国際': 'https://news.google.com/rss/headlines/section/topic/WORLD?hl=ja&gl=JP&ceid=JP:ja', # 4つ目のカテゴリ
}

# --- Flask App Setup ---
app = Flask(__name__)
# 開発環境でフロントエンドとバックエンドが異なるポートで動くためCORSを許可
CORS(app) 

@app.route('/api/news/<category>', methods=['GET'])
def get_news(category):
    """
    指定されたカテゴリのニュースを取得し、Gemini APIでタグ付けしてJSONで返す
    """
    feed_url = RSS_FEEDS.get(category)
    if not feed_url:
        return jsonify({"error": "Invalid category"}), 400

    print(f"Fetching news for category: {category} from {feed_url}")
    
    try:
        # RSSフィードの取得
        response = requests.get(feed_url, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        news_list = []
        
        # Google Newsの場合、item要素は channel/item の下にあることが多い
        # ElementTreeのXPathで全てのitem要素を取得
        for i, item in enumerate(root.findall('.//item')):
            # 最低10件表示の要件を満たすため、15件まで処理
            if len(news_list) >= 15:
                break

            title = item.find('title').text if item.find('title') is not None else "No Title"
            link = item.find('link').text if item.find('link') is not None else "#"
            # ニュースの説明文を取得（ない場合はタイトルを代わりに使用）
            description_element = item.find('description')
            description = description_element.text if description_element is not None else title
            
            # --- Gemini APIによるタグ生成 (時間がかかる部分) ---
            # ニュースのタイトルと説明を結合してプロンプトにする
            news_content = f"記事のタイトル: {title}\n記事の概要: {description}"
            
            system_prompt = """
                # 命令
                提供されたニュース記事の内容を分析し、記事を最もよく表すタグを5つ程度、日本語で生成してください。
                # 制約条件
                必ず「,（カンマ）」区切りで、タグ名のみを出力してください。
                タグには「#」を付けないでください。
                # 出力形式例
                テクノロジー,AI,新製品,モバイル,ニュース
            """
            
            request_prompt = generate_request_prompt(system_prompt, news_content, 0.5, 1.0)
            tags = chat(request_prompt) # Gemini API呼び出し
            
            # 結果をリストに追加
            news_list.append({
                'id': i,
                'title': title,
                'link': link,
                'description': description,
                'tags': tags, # Gemini APIが生成したタグ
            })
            print(f"Processed article {i+1}: {title}")


        if len(news_list) < 10:
             print(f"警告: 取得できた記事は{len(news_list)}件でした。最低10件の要件を満たしていません。")
             
        return jsonify(news_list)

    except requests.exceptions.RequestException as e:
        print(f"RSS取得中にエラーが発生しました: {e}")
        return jsonify({"error": f"Failed to fetch RSS feed: {str(e)}"}), 500
    except ET.ParseError as e:
        print(f"XMLパースエラー: {e}")
        return jsonify({"error": f"Failed to parse RSS feed: {str(e)}"}), 500

if __name__ == '__main__':
    # 起動前に環境変数 GEMINI_API_KEY が設定されていることを確認してください
    if not GEMINI_API_KEY:
        print("エラー: 実行前に GEMINI_API_KEY 環境変数を設定してください。")
    else:
        # 実行: http://127.0.0.1:5000/ にて起動
        app.run(debug=True, port=5000)