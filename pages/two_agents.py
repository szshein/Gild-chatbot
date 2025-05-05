import streamlit as st
from openai import OpenAI
import time
import re
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Import ConversableAgent class
import autogen
from autogen import ConversableAgent, LLMConfig, Agent
from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from autogen.code_utils import content_str
from coding.constant import JOB_DEFINITION, RESPONSE_FORMAT

# --- 中文字型設定 ---
font_path = "msyh.ttc"  # 確保這個字型檔案存在於你的專案目錄或系統路徑中
try:
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
except FileNotFoundError:
    st.warning(f"警告：找不到字型檔案 '{font_path}'。中文顯示可能會異常。")

# Load environment variables from .env file
load_dotenv(override=True)

# URL configurations
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', None)
OPEN_API_KEY = os.getenv('OPEN_API_KEY', None)

placeholderstr = "請輸入你的職缺或技能"
user_name = "Gild"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

seed = 42

llm_config_gemini = LLMConfig(
    api_type = "google",
    model="gemini-2.0-flash-lite",
    api_key=GEMINI_API_KEY,
)

llm_config_openai = LLMConfig(
    api_type = "openai",
    model="gpt-4o-mini",
    api_key=OPEN_API_KEY,
)

def extract_content_and_qualification(text):
    pattern_content = re.compile(r'(?:內容|職責|do(?:[】•])*)(.*?)(?=(?:資格|待遇|報名|期間|條件|$))', re.DOTALL)
    match_content = pattern_content.search(text)
    jobContent = match_content.group(1).strip() if match_content else text.strip()

    pattern_qualification = re.compile(r'(?:條件|資格)[：:](.*?)(?=(?:工作地點|待遇|條件|要求|$))', re.DOTALL)
    match_qualification = pattern_qualification.search(text)
    jobQualification = match_qualification.group(1).strip() if match_qualification else ""

    return pd.Series([jobContent, jobQualification], index=["jobContent", "jobQualification"])

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.05)

def save_lang():
    st.session_state['lang_setting'] = st.session_state.get("language_select")

def paging():
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/two_agents.py", label="Two Agents' Talk", icon="💭")

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
        },
        page_icon="img/favicon.ico"
    )

    # Show title and description.
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        paging()

        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=0, on_change=save_lang, key="language_select")
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")

    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    # --- 載入與前處理資料 ---
    try:
        df = pd.read_csv("104_jobs_all.csv")
        df = df.dropna(subset=['jobName', 'description', 'jobAddrNoDesc'])
        df[['jobContent', 'jobQualification']] = df['description'].apply(extract_content_and_qualification)
        # st.dataframe(df.head()) # 可以選擇顯示部分資料看看
    except FileNotFoundError:
        st.error("找不到 '104_jobs_all.csv' 檔案，請確認檔案已放置在正確的路徑。")
        return

    # 新增一個 radio button 讓使用者選擇輸入類型
    input_type = st.radio(
        "你想輸入的是？",
        ("感興趣的職缺", "你的技能"),
        horizontal=True,
        key="input_type"
    )

    # --- 設定 Agent ---
    with llm_config_gemini:
        student_agent = ConversableAgent(
            name="Student_Agent",
            system_message="你是一位積極的學生，正在尋找實習職缺。你會根據使用者的輸入（可能是感興趣的職缺類型或擁有的技能），從提供的職缺資料中尋找相關的資訊，並向實習職缺推薦老師尋求建議。",
        )
        teacher_agent = ConversableAgent(
            name="Teacher_Agent",
            system_message=(
                "你是一位經驗豐富的實習職缺推薦老師。你會根據學生的需求（他們感興趣的職缺類型或擁有的技能），從提供的職缺資料中找出相關的職缺，並提供職缺名稱、工作內容和職位資格等資訊。"
            ),
        )

    def generate_response(prompt):
        # 在這裡，你可以讓 student_agent 根據 prompt 和 df 的內容進行初步的篩選或總結
        response = student_agent.initiate_chat(
            teacher_agent,
            message = prompt,
            summary_method="reflection_with_llm",
            max_turns=2,
        )
        return response.chat_history

    def show_chat_history(chat_hsitory):
        for entry in chat_hsitory:
            role = entry.get('role')
            name = entry.get('name')
            content = entry.get('content')
            st.session_state.messages.append({"role": f"{role}", "content": content})

            if len(content.strip()) != 0:
                if 'ALL DONE' in content:
                    return
                else:
                    if role != 'assistant':
                        st_c_chat.chat_message(f"{role}").write((content))
                    else:
                        st_c_chat.chat_message("user",avatar=user_image).write(content)
        return

    def chat(prompt: str):
        response = generate_response(prompt)
        show_chat_history(response)

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        if st.session_state.input_type == "感興趣的職缺":
            chat(f"我對 {prompt} 的實習職缺感興趣。請根據職缺資料提供相關的資訊。")
        elif st.session_state.input_type == "你的技能":
            chat(f"我的技能包括：{prompt}。請根據我的技能，從職缺資料中推薦合適的實習機會。")

if __name__ == "__main__":
    main()