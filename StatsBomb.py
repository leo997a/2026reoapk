import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from mplsoccer import Pitch, VerticalPitch
from highlight_text import fig_text, ax_text
from PIL import Image
from urllib.request import urlopen
from unidecode import unidecode
from scipy.spatial import ConvexHull
import arabic_reshaper
from bidi.algorithm import get_display
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import warnings
import os
import requests
from io import StringIO, BytesIO
import matplotlib.font_manager as fm
import ast


# دالة إضافة العلامة المائية
def add_watermark(fig, text="reo show", alpha=0.3, fontsize=15, color='white', x_pos=0.5, y_pos=0.5, ha='center', va='center'):
    """
    إضافة علامة مائية إلى الرسم البياني
    
    المعلمات:
    fig : matplotlib.figure.Figure
        الرسم البياني المراد إضافة العلامة المائية إليه
    text : str
        نص العلامة المائية (اسم القناة)
    alpha : float
        شفافية العلامة المائية (0-1)
    fontsize : int
        حجم خط العلامة المائية
    color : str
        لون العلامة المائية
    x_pos : float
        موضع العلامة المائية على المحور الأفقي (0-1)
    y_pos : float
        موضع العلامة المائية على المحور الرأسي (0-1)
    ha : str
        محاذاة أفقية ('left', 'center', 'right')
    va : str
        محاذاة رأسية ('top', 'center', 'bottom')
    """
    # إضافة نص العلامة المائية في الموضع المحدد
    fig.text(x_pos, y_pos, text, 
             fontsize=fontsize, 
             color=color,
             ha=ha,
             va=va,
             alpha=alpha,
             transform=fig.transFigure,
             fontweight='bold',
             path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    return fig

# إضافة الخطوط يدويًا من مجلد fonts
font_dir = os.path.join(os.getcwd(), 'fonts')
if os.path.exists(font_dir):
    for font_file in os.listdir(font_dir):
        if font_file.endswith('.ttf') or font_file.endswith('.otf'):
            font_path = os.path.join(font_dir, font_file)
            fm.fontManager.addfont(font_path)
            st.write(f"تم تحميل الخط: {font_file}")
else:
    st.warning("مجلد 'fonts' غير موجود. تأكد من إضافة ملفات الخطوط.")

# الحصول على قائمة الخطوط المتوفرة
available_fonts = [f.name for f in fm.fontManager.ttflist]

# تصفية الخطوط ذات الصلة
filtered_fonts = [font for font in available_fonts if 'Arabic' in font or 'DejaVu' in font or 'Tajawal' in font or 'Cairo' in font]
st.write("الخطوط المتوفرة (مرشحة):", filtered_fonts)

# الخطوط المتاحة للاختيار
font_options = ['Tajawal', 'Cairo', 'Noto Sans Arabic', 'DejaVu Sans']
available_font_options = [font for font in font_options if font in available_fonts]

if not available_font_options:
    st.error("لم يتم العثور على أي خطوط مدعومة. تأكد من إضافة الخطوط إلى مجلد 'fonts'.")
    available_font_options = ['DejaVu Sans']  # خط افتراضي

warnings.filterwarnings("ignore", category=DeprecationWarning)

# قراءة ملف الفرق من الرابط
url = "https://raw.githubusercontent.com/leo997a/2026reoapk/refs/heads/main/teams_name_and_id.csv"
try:
    response = requests.get(url)
    response.raise_for_status()
    teams_df = pd.read_csv(StringIO(response.text))
    fotmob_team_ids = dict(zip(teams_df['teamName'], teams_df['teamId']))
except requests.exceptions.RequestException as e:
    st.error(f"خطأ أثناء تحميل ملف teams_name_and_id.csv من الرابط: {e}")
    fotmob_team_ids = {}
except Exception as e:
    st.error(f"خطأ أثناء قراءة الملف: {e}")
    fotmob_team_ids = {}

# دالة لتحويل النص العربي


def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


# إضافة CSS لدعم RTL
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stSelectbox > div > div > div {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# إعداد الخطوط الحديثة (يجب تثبيت الخط أو استخدام خط متاح)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'Montserrat', 'Arial']

# تعريف الألوان الافتراضية
default_hcol = '#d00000'
default_acol = '#003087'
default_bg_color = '#1e1e2f'
default_gradient_colors = ['#003087', '#d00000']
violet = '#800080'

# إعداد Session State
if 'json_data' not in st.session_state:
    st.session_state.json_data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'teams_dict' not in st.session_state:
    st.session_state.teams_dict = None
if 'players_df' not in st.session_state:
    st.session_state.players_df = None
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False

# إضافة أدوات اختيار الألوان في الشريط الجانبي
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker(
    'لون الفريق المضيف',
    default_hcol,
    key='hcol_picker')
acol = st.sidebar.color_picker(
    'لون الفريق الضيف',
    default_acol,
    key='acol_picker')
bg_color = st.sidebar.color_picker(
    'لون الخلفية',
    default_bg_color,
    key='bg_color_picker')
gradient_start = st.sidebar.color_picker(
    'بداية التدرج',
    default_gradient_colors[0],
    key='gradient_start_picker')
gradient_end = st.sidebar.color_picker(
    'نهاية التدرج',
    default_gradient_colors[1],
    key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker(
    'لون الخطوط', '#ffffff', key='line_color_picker')
selected_font = st.sidebar.selectbox("اختر الخط للنصوص:", available_font_options, index=0)

# إعداد تدرج لوني للأشرطة
cmap = LinearSegmentedColormap.from_list("custom_gradient", gradient_colors)
# إضافة إعدادات العلامة المائية
st.sidebar.title('إعدادات العلامة المائية')
watermark_enabled = st.sidebar.checkbox('تفعيل العلامة المائية', value=True)
watermark_text = st.sidebar.text_input('نص العلامة المائية', value='reo show')
watermark_opacity = st.sidebar.slider('شفافية العلامة المائية', min_value=0.1, max_value=1.0, value=0.3, step=0.1)
watermark_size = st.sidebar.slider('حجم العلامة المائية', min_value=8, max_value=30, value=15, step=1)
watermark_color = st.sidebar.color_picker('لون العلامة المائية', value='#FFFFFF')

# إضافة خيارات موضع العلامة المائية
st.sidebar.subheader('موضع العلامة المائية')
watermark_position = st.sidebar.selectbox(
    'موضع العلامة المائية',
    ['وسط', 'أعلى اليمين', 'أعلى اليسار', 'أسفل اليمين', 'أسفل اليسار', 'مخصص'],
    index=0
)

# تعيين قيم الموضع بناءً على الاختيار
if watermark_position == 'وسط':
    watermark_x = 0.5
    watermark_y = 0.5
    watermark_ha = 'center'
    watermark_va = 'center'
elif watermark_position == 'أعلى اليمين':
    watermark_x = 0.95
    watermark_y = 0.95
    watermark_ha = 'right'
    watermark_va = 'top'
elif watermark_position == 'أعلى اليسار':
    watermark_x = 0.05
    watermark_y = 0.95
    watermark_ha = 'left'
    watermark_va = 'top'
elif watermark_position == 'أسفل اليمين':
    watermark_x = 0.95
    watermark_y = 0.05
    watermark_ha = 'right'
    watermark_va = 'bottom'
elif watermark_position == 'أسفل اليسار':
    watermark_x = 0.05
    watermark_y = 0.05
    watermark_ha = 'left'
    watermark_va = 'bottom'
else:  # مخصص
    watermark_x = st.sidebar.slider('الموضع الأفقي (X)', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    watermark_y = st.sidebar.slider('الموضع الرأسي (Y)', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    watermark_ha = st.sidebar.selectbox('المحاذاة الأفقية', ['left', 'center', 'right'], index=1)
    watermark_va = st.sidebar.selectbox('المحاذاة الرأسية', ['top', 'center', 'bottom'], index=1)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# دالة استخراج البيانات من WhoScored


@st.cache_data
def extract_match_dict_from_html(uploaded_file):
    try:
        # قراءة محتوى ملف HTML
        html_content = uploaded_file.read().decode('utf-8')
        
        # تسجيل أول 1000 وآخر 1000 حرف من HTML للتحقق
        st.write("HTML content preview (first 1000 chars):", html_content[:1000])
        st.write("HTML content preview (last 1000 chars):", html_content[-1000:])
        with open("html_content.txt", "w", encoding="utf-8") as f:
            f.write(html_content)
        st.write("تم حفظ محتوى HTML الكامل في html_content.txt")
        
        # استخراج نص JSON باستخدام تعبير منتظم
        regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
        match = re.search(regex_pattern, html_content)
        if not match:
            st.error("لم يتم العثور على require.config.params[\"args\"] في ملف HTML")
            return None
        
        data_txt = match.group(0)
        
        # تنظيف النص لتحويله إلى JSON صالح
        data_txt = data_txt.replace('matchId', '"matchId"')
        data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
        data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
        data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
        data_txt = data_txt.replace('};', '}')
        
        # تسجيل معاينة للنص المستخرج
        st.write("JSON string preview (first 500 chars):", data_txt[:500])
        st.write("JSON string preview (last 500 chars):", data_txt[-500:])
        with open("raw_json.txt", "w", encoding="utf-8") as f:
            f.write(data_txt)
        st.write("تم حفظ النص المستخرج الخام في raw_json.txt")
        
        # التحقق من وجود المفاتيح المتوقعة
        if '"formationIdNameMappings"' not in data_txt:
            st.error("النص المستخرج غير مكتمل: لا يحتوي على formationIdNameMappings")
            with open("failed_json.txt", "w", encoding="utf-8") as f:
                f.write(data_txt)
            st.write("تم حفظ النص المستخرج في failed_json.txt للتحقق")
            return None
        
        # محاولة تحليل JSON
        try:
            matchdict = json.loads(data_txt)
        except json.JSONDecodeError as json_err:
            st.error(f"خطأ في تحليل JSON: {str(json_err)}")
            with open("failed_json.txt", "w", encoding="utf-8") as f:
                f.write(data_txt)
            st.write("تم حفظ النص المستخرج في failed_json.txt للتحقق")
            return None
        
        # استخراج matchCentreData فقط
        if 'matchCentreData' not in matchdict:
            st.error("لم يتم العثور على matchCentreData في البيانات المحللة")
            return None
        
        return matchdict['matchCentreData']
    
    except Exception as e:
        st.error(f"خطأ أثناء استخراج البيانات من HTML: {str(e)}")
        return None
def get_event_data(json_data):
    events_dict = json_data["events"]
    teams_dict = {json_data['home']['teamId']: json_data['home']['name'],
                  json_data['away']['teamId']: json_data['away']['name']}
    players_dict = json_data["playerIdNameDictionary"]
    
    # إنشاء إطار بيانات اللاعبين
    players_home_df = pd.DataFrame(json_data['home']['players'])
    players_home_df["teamId"] = json_data['home']['teamId']
    players_away_df = pd.DataFrame(json_data['away']['players'])
    players_away_df["teamId"] = json_data['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    players_df['name'] = players_df['name'].astype(str)
    players_df['name'] = players_df['name'].apply(unidecode)
    
    df = pd.DataFrame(events_dict)
    dfp = pd.DataFrame(players_df)
    
    # استخراج displayName من الأنواع
    df['type'] = df['type'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else str(x))
    df['outcomeType'] = df['outcomeType'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else str(x))
    df['period'] = df['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else str(x))

    # تسجيل قيم period الأصلية
    st.write("قيم period الأصلية:", df['period'].unique())    
    # تحويل قيم period إلى نصوص موحدة
    period_mapping = {
        'FirstHalf': 'FirstHalf', 'SecondHalf': 'SecondHalf',
        'FirstPeriodOfExtraTime': 'FirstPeriodOfExtraTime',
        'SecondPeriodOfExtraTime': 'SecondPeriodOfExtraTime',
        'PenaltyShootout': 'PenaltyShootout', 'PostGame': 'PostGame',
        'PreMatch': 'PreMatch',
        # التعامل مع القيم الرقمية أو غير المتوقعة
        1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime',
        4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame',
        16: 'PreMatch'
    }
    df['period'] = df['period'].map(period_mapping).fillna(df['period'])    
    
    df['period'] = df['period'].replace({
        'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3,
        'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16
    }).infer_objects(copy=False)
    
    def cumulative_match_mins(events_df):
        events_out = pd.DataFrame()
        match_events = events_df.copy()
        match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
        for period in np.arange(1, match_events['period'].max() + 1, 1):
            if period > 1:
                t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                          match_events[match_events['period'] == period]['cumulative_mins'].min()
            else:
                t_delta = 0
            match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
        events_out = pd.concat([events_out, match_events])
        return events_out
    
    df = cumulative_match_mins(df)
    
    def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50):
        events_out = pd.DataFrame()
        min_carry_length = 3.0
        max_carry_length = 100.0
        min_carry_duration = 1.0
        max_carry_duration = 50.0
        match_events = events_df.reset_index()
        match_events.loc[match_events['type'] == 'BallRecovery', 'endX'] = match_events.loc[match_events['type'] == 'BallRecovery', 'endX'].fillna(match_events['x'])
        match_events.loc[match_events['type'] == 'BallRecovery', 'endY'] = match_events.loc[match_events['type'] == 'BallRecovery', 'endY'].fillna(match_events['y'])
        match_carries = pd.DataFrame()
        
        for idx, match_event in match_events.iterrows():
            if idx < len(match_events) - 1:
                prev_evt_team = match_event['teamId']
                next_evt_idx = idx + 1
                init_next_evt = match_events.loc[next_evt_idx]
                take_ons = 0
                incorrect_next_evt = True
                while incorrect_next_evt:
                    next_evt = match_events.loc[next_evt_idx]
                    if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                        take_ons += 1
                        incorrect_next_evt = True
                    elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['type'] == 'Foul')
                          or (next_evt['type'] == 'Card')):
                        incorrect_next_evt = True
                    else:
                        incorrect_next_evt = False
                    next_evt_idx += 1
                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['type'] != 'BallTouch'
                dx = 105 * (match_event['endX'] - next_evt['x']) / 100
                dy = 68 * (match_event['endY'] - next_evt['y']) / 100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                min_time = dt >= min_carry_duration
                same_phase = dt < max_carry_duration
                same_period = match_event['period'] == next_evt['period']
                valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase & same_period
                if valid_carry:
                    carry = pd.DataFrame()
                    prev = match_event
                    nex = next_evt
                    carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                    carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                        prev['minute'] * 60 + prev['second'])) / (2 * 60))
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                        prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) + (
                        prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = 'Carry'
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                    carry['isTouch'] = True
                    carry['playerId'] = nex['playerId']
                    carry['endX'] = nex['x']
                    carry['endY'] = nex['y']
                    carry['blockedX'] = np.nan
                    carry['blockedY'] = np.nan
                    carry['goalMouthZ'] = np.nan
                    carry['goalMouthY'] = np.nan
                    carry['isShot'] = np.nan
                    carry['relatedEventId'] = nex['eventId']
                    carry['relatedPlayerId'] = np.nan
                    carry['isGoal'] = np.nan
                    carry['cardType'] = np.nan
                    carry['isOwnGoal'] = np.nan
                    carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2
                    match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)
        match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
        match_events_and_carries = match_events_and_carries.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)
        events_out = pd.concat([events_out, match_events_and_carries])
        return events_out
    
    df = insert_ball_carries(df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50)
    df = df.reset_index(drop=True)
    df['index'] = range(1, len(df) + 1)
    df = df[['index'] + [col for col in df.columns if col != 'index']]
    
    # Assign xT values
    df_base = df
    dfxT = df_base.copy()
    dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
    dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
    dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType'] == 'Successful')]
    
    # تحميل xT_Grid.csv
    xT = pd.read_csv("https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/refs/heads/main/xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape
    
    dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
    dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
    dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
    dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)
    
    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x.iloc[1]][x.iloc[0]], axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x.iloc[1]][x.iloc[0]], axis=1)
    
    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    columns_to_drop = [
        'eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period', 'outcomeType', 'qualifiers', 'type',
        'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY', 'relatedEventId', 'relatedPlayerId', 'blockedX',
        'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins'
    ]
    dfxT.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    df = df.merge(dfxT, on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1-i] for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)
    
    # تحجيم البيانات إلى 105x68
    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05
    df['endY'] = df['endY'] * 0.68
    df['goalMouthY'] = df['goalMouthY'] * 0.68
    # تنظيف الإحداثيات
    for col in ['x', 'y', 'endX', 'endY', 'goalMouthY']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if col in ['x', 'endX']:
            df[col] = df[col].clip(0, 105)  # تحديد النطاق بين 0 و105
        elif col in ['y', 'endY', 'goalMouthY']:
            df[col] = df[col].clip(0, 68)   # تحديد النطاق بين 0 و68
    
    # تسجيل الإحداثيات بعد التنظيف
    st.write("إحصائيات الإحداثيات بعد التنظيف:")
    for col in ['x', 'y', 'endX', 'endY']:
        st.write(f"{col} - min: {df[col].min()}, max: {df[col].max()}, NaN count: {df[col].isna().sum()}")    
    
    columns_to_drop = [
        'height', 'weight', 'age', 'isManOfTheMatch', 'field', 'stats', 'subbedInPlayerId', 'subbedOutPeriod',
        'subbedOutExpandedMinute', 'subbedInPeriod', 'subbedInExpandedMinute', 'subbedOutPlayerId', 'teamId'
    ]
    dfp.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df.merge(dfp, on='playerId', how='left')
    
    df['qualifiers'] = df['qualifiers'].astype(str)
    df['prog_pass'] = np.where((df['type'] == 'Pass'), 
                               np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'), 
                                np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
    
    df['name'] = df['name'].astype(str)
    df['name'] = df['name'].apply(unidecode)
    
    def get_short_name(full_name):
        if pd.isna(full_name):
            return full_name
        parts = full_name.split()
        if len(parts) == 1:
            return full_name
        elif len(parts) == 2:
            return parts[0][0] + ". " + parts[1]
        else:
            return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])
    
    df['shortName'] = df['name'].apply(get_short_name)
    columns_to_drop2 = ['id']
    df.drop(columns=columns_to_drop2, inplace=True, errors='ignore')
    
    return df, teams_dict, players_df

    def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
        events_out = pd.DataFrame()
        match_events_df = events_df.reset_index()
        match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded', 'Start', 'Card', 'SubstitutionOff',
                                                                             'SubstitutionOn', 'FormationChange', 'FormationSet', 'End'])].copy()
        match_pos_events_df['outcomeBinary'] = (
            match_pos_events_df['outcomeType'] .apply(
                lambda x: 1 if x == 'Successful' else 0))
        match_pos_events_df['teamBinary'] = (
            match_pos_events_df['teamName'] .apply(
                lambda x: 1 if x == min(
                    match_pos_events_df['teamName']) else 0))
        match_pos_events_df['goalBinary'] = (
            (match_pos_events_df['type'] == 'Goal') .astype(int).diff(
                periods=1).apply(
                lambda x: 1 if x < 0 else 0))
        pos_chain_df = pd.DataFrame()
        for n in np.arange(1, chain_check):
            pos_chain_df[f'evt_{n}_same_team'] = abs(
                match_pos_events_df['teamBinary'].diff(periods=-n))
            pos_chain_df[f'evt_{n}_same_team'] = pos_chain_df[f'evt_{n}_same_team'].apply(
                lambda x: 1 if x > 1 else x)
        pos_chain_df['enough_evt_same_team'] = pos_chain_df.sum(axis=1).apply(
            lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
        pos_chain_df['enough_evt_same_team'] = pos_chain_df['enough_evt_same_team'].diff(
            periods=1)
        pos_chain_df[pos_chain_df['enough_evt_same_team'] < 0] = 0
        match_pos_events_df['period'] = pd.to_numeric(
            match_pos_events_df['period'], errors='coerce')
        pos_chain_df['upcoming_ko'] = 0
        for ko in match_pos_events_df[(match_pos_events_df['goalBinary'] == 1) | (
                match_pos_events_df['period'].diff(periods=1))].index.values:
            ko_pos = match_pos_events_df.index.to_list().index(ko)
            pos_chain_df.iloc[ko_pos - suc_evts_in_chain:ko_pos,
                              pos_chain_df.columns.get_loc('upcoming_ko')] = 1
        pos_chain_df['valid_pos_start'] = (
            pos_chain_df.fillna(0)['enough_evt_same_team'] -
            pos_chain_df.fillna(0)['upcoming_ko'])
        pos_chain_df['kick_off_period_change'] = match_pos_events_df['period'].diff(
            periods=1)
        pos_chain_df['kick_off_goal'] = (
            (match_pos_events_df['type'] == 'Goal') .astype(int).diff(
                periods=1).apply(
                lambda x: 1 if x < 0 else 0))
        pos_chain_df.loc[pos_chain_df['kick_off_period_change']
                         == 1, 'valid_pos_start'] = 1
        pos_chain_df.loc[pos_chain_df['kick_off_goal']
                         == 1, 'valid_pos_start'] = 1
        pos_chain_df['teamName'] = match_pos_events_df['teamName']
        pos_chain_df.loc[pos_chain_df.head(1).index, 'valid_pos_start'] = 1
        pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_id'] = 1
        pos_chain_df.loc[pos_chain_df.head(
            1).index, 'possession_team'] = pos_chain_df.loc[pos_chain_df.head(1).index, 'teamName']
        valid_pos_start_id = pos_chain_df[pos_chain_df['valid_pos_start'] > 0].index
        possession_id = 2
        for idx in np.arange(1, len(valid_pos_start_id)):
            current_team = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
            previous_team = pos_chain_df.loc[valid_pos_start_id[idx - 1], 'teamName']
            if ((previous_team == current_team) & (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_goal'] != 1) &
                    (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_period_change'] != 1)):
                pos_chain_df.loc[valid_pos_start_id[idx],
                                 'possession_id'] = np.nan
            else:
                pos_chain_df.loc[valid_pos_start_id[idx],
                                 'possession_id'] = possession_id
                pos_chain_df.loc[valid_pos_start_id[idx],
                                 'possession_team'] = pos_chain_df.loc[valid_pos_start_id[idx],
                                                                       'teamName']
                possession_id += 1
        match_events_df = pd.merge(match_events_df,
                                   pos_chain_df[['possession_id',
                                                 'possession_team']],
                                   how='left',
                                   left_index=True,
                                   right_index=True)
        match_events_df[['possession_id', 'possession_team']] = (
            match_events_df[['possession_id', 'possession_team']].fillna(method='ffill'))
        match_events_df[['possession_id', 'possession_team']] = (
            match_events_df[['possession_id', 'possession_team']].fillna(method='bfill'))
        events_out = pd.concat([events_out, match_events_df])
        return events_out

    df = get_possession_chains(df, 5, 3)
    df['period'] = df['period'].replace({1: 'FirstHalf',
                                         2: 'SecondHalf',
                                         3: 'FirstPeriodOfExtraTime',
                                         4: 'SecondPeriodOfExtraTime',
                                         5: 'PenaltyShootout',
                                         14: 'PostGame',
                                         16: 'PreMatch'})
    df = df[df['period'] != 'PenaltyShootout']
    df = df.reset_index(drop=True)
    return df, teams_dict, players_df
    
# دالة لإعادة تشكيل النصوص العربية
def pass_network(ax, team_name, col, phase_tag, hteamName, ateamName, hgoal_count, agoal_count, hteamID, ateamID):
    try:
        # تسجيل قيم period الفريدة للتصحيح
        st.write("قيم period الفريدة في st.session_state.df:", st.session_state.df['period'].unique())
        
        # تصفية البيانات بناءً على phase_tag
        if phase_tag == 'Full Time':
            df_pass = st.session_state.df.copy()
        elif phase_tag == 'First Half':
            df_pass = st.session_state.df[st.session_state.df['period'].isin(['FirstHalf', 1])]
        elif phase_tag == 'Second Half':
            df_pass = st.session_state.df[st.session_state.df['period'].isin(['SecondHalf', 2])]
        else:
            raise ValueError(f"Invalid phase_tag: {phase_tag}")
        
        df_pass = df_pass.reset_index(drop=True)
        
        # التحقق من وجود بيانات
        if df_pass.empty:
            ax.text(34, 60, reshape_arabic_text(f"لا توجد بيانات لـ {phase_tag}"), 
                    color='white', fontsize=14, ha='center', va='center', weight='bold')
            st.warning(f"لا توجد بيانات لـ {phase_tag}. تحقق من قيم عمود 'period'.")
            return pd.DataFrame()
        
        # تصفية التمريرات
        total_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass')]
        accrt_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful')]
        if len(total_pass) == 0:
            ax.text(34, 60, reshape_arabic_text(f"لا توجد تمريرات لـ {team_name} في {phase_tag}"), 
                    color='white', fontsize=14, ha='center', va='center', weight='bold')
            st.warning(f"لا توجد تمريرات لـ {team_name} في {phase_tag}.")
            return pd.DataFrame()
        
        # حساب دقة التمريرات
        accuracy = round((len(accrt_pass) / len(total_pass)) * 100, 2) if len(total_pass) != 0 else 0
        
        # إضافة عمود pass_receiver
        df_pass['pass_receiver'] = df_pass.loc[(df_pass['type'] == 'Pass') & 
                                               (df_pass['outcomeType'] == 'Successful') & 
                                               (df_pass['teamName'].shift(-1) == team_name), 'name'].shift(-1)
        df_pass['pass_receiver'] = df_pass['pass_receiver'].fillna('No')
        
        # تصفية الأحداث الهجومية
        off_acts_df = df_pass[(df_pass['teamName'] == team_name) & 
                              (df_pass['type'].isin(['Pass', 'Goal', 'MissedShots', 'SavedShot', 'ShotOnPost', 'TakeOn', 'BallTouch', 'KeeperPickup']))]
        off_acts_df = off_acts_df[['name', 'x', 'y']].reset_index(drop=True)
        if off_acts_df.empty:
            ax.text(34, 60, reshape_arabic_text(f"لا توجد أحداث هجومية لـ {team_name} في {phase_tag}"), 
                    color='white', fontsize=14, ha='center', va='center', weight='bold')
            st.warning(f"لا توجد أحداث هجومية لـ {team_name} في {phase_tag}.")
            return pd.DataFrame()
        
        # حساب متوسط المواقع
        avg_locs_df = off_acts_df.groupby('name').agg(avg_x=('x', 'median'), avg_y=('y', 'median')).reset_index()
        team_pdf = st.session_state.players_df[['name', 'shirtNo', 'position', 'isFirstEleven']].copy()
        if team_pdf.empty:
            ax.text(34, 60, reshape_arabic_text("بيانات اللاعبين غير متاحة"), 
                    color='white', fontsize=14, ha='center', va='center', weight='bold')
            st.warning("بيانات اللاعبين غير متاحة.")
            return pd.DataFrame()
        
        # دمج بيانات اللاعبين
        avg_locs_df = avg_locs_df.merge(team_pdf, on='name', how='left')
        avg_locs_df['isFirstEleven'] = avg_locs_df['isFirstEleven'].fillna(False).infer_objects(copy=False)
        avg_locs_df['shirtNo'] = avg_locs_df['shirtNo'].fillna(0)
        avg_locs_df['position'] = avg_locs_df['position'].fillna('Unknown')
        
        # تصفية التمريرات الناجحة (باستثناء الركنيات والركلات الحرة)
        df_pass = df_pass[(df_pass['type'] == 'Pass') & 
                          (df_pass['outcomeType'] == 'Successful') & 
                          (df_pass['teamName'] == team_name) & 
                          (~df_pass['qualifiers'].str.contains('Corner|Freekick', na=False))]
        if len(df_pass) == 0:
            ax.text(34, 60, reshape_arabic_text(f"لا توجد تمريرات ناجحة لـ {team_name} في {phase_tag}"), 
                    color='white', fontsize=14, ha='center', va='center', weight='bold')
            st.warning(f"لا توجد تمريرات ناجحة لـ {team_name} في {phase_tag}.")
            return pd.DataFrame()
        
        df_pass = df_pass[['type', 'name', 'pass_receiver']].reset_index(drop=True)
        pass_count_df = df_pass.groupby(['name', 'pass_receiver']).size().reset_index(name='pass_count').sort_values(by='pass_count', ascending=False)
        pass_count_df = pass_count_df.reset_index(drop=True)
        
        # إنشاء إطار بيانات التمريرات
        pass_counts_df = pd.merge(pass_count_df, avg_locs_df, on='name', how='left')
        pass_counts_df.rename(columns={'avg_x': 'pass_avg_x', 'avg_y': 'pass_avg_y'}, inplace=True)
        pass_counts_df = pd.merge(pass_counts_df, avg_locs_df, left_on='pass_receiver', right_on='name', how='left', suffixes=('', '_receiver'))
        pass_counts_df.drop(columns=['name_receiver'], inplace=True)
        pass_counts_df.rename(columns={'avg_x': 'receiver_avg_x', 'avg_y': 'receiver_avg_y'}, inplace=True)
        pass_counts_df = pass_counts_df.sort_values(by='pass_count', ascending=False).reset_index(drop=True)
        pass_counts_df = pass_counts_df.dropna(subset=['shirtNo_receiver'])
        
        if pass_counts_df.empty:
            ax.text(34, 60, reshape_arabic_text(f"لا توجد تمريرات بين اللاعبين لـ {team_name} في {phase_tag}"), 
                    color='white', fontsize=14, ha='center', va='center', weight='bold')
            st.warning(f"لا توجد تمريرات بين اللاعبين لـ {team_name} في {phase_tag}.")
            return pd.DataFrame()
        
        # إنشاء إطار بيانات pass_btn
        pass_btn = pass_counts_df[['name', 'shirtNo', 'pass_receiver', 'shirtNo_receiver', 'pass_count']]
        pass_btn.loc[:, 'shirtNo_receiver'] = pass_btn['shirtNo_receiver'].astype(float).astype(int)
        
        # إعداد الرسم
        MAX_LINE_WIDTH = 8
        MIN_LINE_WIDTH = 0.5
        MIN_TRANSPARENCY = 0.2
        MAX_TRANSPARENCY = 0.9
        pass_counts_df['line_width'] = (pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()) * (MAX_LINE_WIDTH - MIN_LINE_WIDTH) + MIN_LINE_WIDTH
        c_transparency = pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()
        c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
        color = np.array(to_rgba(col))
        color = np.tile(color, (len(pass_counts_df), 1))
        color[:, 3] = c_transparency
        
        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, linewidth=1.5, line_color=line_color)
        pitch.draw(ax=ax)
        
        gradient = LinearSegmentedColormap.from_list("pitch_gradient", gradient_colors, N=100)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = Y
        ax.imshow(Z, extent=[0, 68, 0, 105], cmap=gradient, alpha=0.8, aspect='auto', zorder=0)
        pitch.draw(ax=ax)
        
        # رسم خطوط التمريرات
        for idx in range(len(pass_counts_df)):
            pitch.lines(
                pass_counts_df['pass_avg_x'].iloc[idx],
                pass_counts_df['pass_avg_y'].iloc[idx],
                pass_counts_df['receiver_avg_x'].iloc[idx],
                pass_counts_df['receiver_avg_y'].iloc[idx],
                lw=pass_counts_df['line_width'].iloc[idx],
                color=color[idx],
                zorder=1,
                ax=ax
            )
        
        # رسم مواقع اللاعبين
        for index, row in avg_locs_df.iterrows():
            if row['isFirstEleven']:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='o', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.9, ax=ax)
            else:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='s', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.7, ax=ax)
        
        # إضافة أرقام القمصان
        for index, row in avg_locs_df.iterrows():
            player_initials = row["shirtNo"]
            pitch.annotate(player_initials, xy=(row.avg_x, row.avg_y), c='white', ha='center', va='center', size=14, weight='bold', ax=ax)
        
        # إضافة خط متوسط المواقع
        avgph = round(avg_locs_df['avg_x'].median(), 2)
        ax.axhline(y=avgph, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # حساب خط الدفاع والهجوم
        center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
        def_line_h = round(center_backs_height['avg_x'].median(), 2) if not center_backs_height.empty else avgph
        
        Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == True].sort_values(by='avg_x', ascending=False).head(2)
        fwd_line_h = round(Forwards_height['avg_x'].mean(), 2) if not Forwards_height.empty else avgph
        
        ymid = [0, 0, 68, 68]
        xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
        ax.fill(ymid, xmid, col, alpha=0.2)
        v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)
        
        # إضافة النتيجة
        score_text = reshape_arabic_text(f"{hteamName} {hgoal_count} - {agoal_count} {ateamName}")
        score = ax.text(34, 120, score_text, color='white', fontsize=18, ha='center', va='center', weight='bold')
        score.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

        # إضافة شعار الفريق
        try:
            teamID_fotmob = fotmob_team_ids.get(team_name, hteamID if team_name == hteamName else ateamID)
            logo_url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{teamID_fotmob}.png"
            response = requests.get(logo_url)
            response.raise_for_status()
            logo = Image.open(BytesIO(response.content))
            logo = logo.resize((50, 50), Image.Resampling.LANCZOS)
            
            logo_ax = ax.inset_axes([0.45, -0.15, 0.1, 0.1], transform=ax.transAxes)
            logo_ax.imshow(logo)
            logo_ax.axis('off')
            
            title = ax.text(34, 1, reshape_arabic_text(f"شبكة تمريرات {team_name}"), 
                            color='white', fontsize=14, ha='center', va='center', weight='bold')
            title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        except Exception as e:
            st.warning(f"فشل في تحميل شعار {team_name} من FotMob: {str(e)}")
            title = ax.text(34, 1, reshape_arabic_text(f"شبكة تمريرات {team_name}"), 
                            color='white', fontsize=14, ha='center', va='center', weight='bold')
            title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # إضافة نصوص الوقت والإحصائيات
        if phase_tag == 'Full Time':
            time_text = ax.text(34, 115, reshape_arabic_text('الوقت بالكامل: 0-90 دقيقة'), 
                                color='white', fontsize=16, ha='center', va='center', weight='bold')
            time_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            stats_text = ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), 
                                 color='white', fontsize=14, ha='center', va='center')
            stats_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        elif phase_tag == 'First Half':
            time_text = ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), 
                                color='white', fontsize=16, ha='center', va='center', weight='bold')
            time_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            stats_text = ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), 
                                 color='white', fontsize=14, ha='center', va='center')
            stats_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        elif phase_tag == 'Second Half':
            time_text = ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), 
                                color='white', fontsize=16, ha='center', va='center', weight='bold')
            time_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            stats_text = ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), 
                                 color='white', fontsize=14, ha='center', va='center')
            stats_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        compact_text = ax.text(34, -6, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), 
                               color='white', fontsize=14, ha='center', va='center', weight='bold')
        compact_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # تعديل حجم ووضوح أرقام القمصان
        for index, row in avg_locs_df.iterrows():
            player_initials = row["shirtNo"]
            number = pitch.annotate(player_initials, xy=(row.avg_x, row.avg_y), c='white', ha='center', va='center', size=16, weight='bold', ax=ax)
            number.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        return pass_btn
    
    except Exception as e:
        ax.text(34, 60, reshape_arabic_text(f"خطأ: {str(e)}"), 
                color='white', fontsize=14, ha='center', va='center', weight='bold')
        st.error(f"خطأ في شبكة التمريرات: {str(e)}")
        return pd.DataFrame()
# دالة مناطق السيطرة


def team_domination_zones(
        ax,
        phase_tag,
        hteamName,
        ateamName,
        hcol,
        acol,
        bg_color,
        line_color,
        gradient_colors):
    """رسم مناطق السيطرة لكل فريق بناءً على اللمسات في كل منطقة بتصميم عصري."""
    df = st.session_state.df.copy()
    if phase_tag == 'First Half':
        df = df[df['period'] == 'FirstHalf']
    elif phase_tag == 'Second Half':
        df = df[df['period'] == 'SecondHalf']

    # تصفية الأحداث التي تتضمن لمسات مفتوحة (Open-Play Touches)
    df_touches = df[df['isTouch']].copy()

    # التحقق من وجود بيانات لمسات
    if df_touches.empty:
        ax.text(
            52.5,
            34,
            reshape_arabic_text('لا توجد بيانات لمسات متاحة'),
            color='white',
            fontsize=14,
            ha='center',
            va='center',
            weight='bold')
        return

    # تقسيم الملعب إلى شبكة (مثل 5x4)
    x_bins = np.linspace(0, 105, 7)  # 6 أعمدة
    y_bins = np.linspace(0, 68, 6)   # 5 صفوف
    df_touches['x_bin'] = pd.cut(
        df_touches['x'],
        bins=x_bins,
        labels=False,
        include_lowest=True)
    df_touches['y_bin'] = pd.cut(
        df_touches['y'],
        bins=y_bins,
        labels=False,
        include_lowest=True)

    # حساب عدد اللمسات لكل فريق في كل منطقة
    touches_by_team = df_touches.groupby(['teamName', 'x_bin', 'y_bin']).size(
    ).unstack(fill_value=0).stack().reset_index(name='touch_count')

    # إنشاء قاموس لتخزين اللمسات لكل فريق
    hteam_touches = touches_by_team[touches_by_team['teamName'] == hteamName].pivot(
        index='y_bin', columns='x_bin', values='touch_count').fillna(0)
    ateam_touches = touches_by_team[touches_by_team['teamName'] == ateamName].pivot(
        index='y_bin', columns='x_bin', values='touch_count').fillna(0)

    # محاذاة المصفوفات للتأكد من أن لديهم نفس الأبعاد
    hteam_touches = hteam_touches.reindex(
        index=range(5),
        columns=range(6),
        fill_value=0)  # 5 صفوف، 6 أعمدة
    ateam_touches = ateam_touches.reindex(
        index=range(5), columns=range(6), fill_value=0)

    # حساب النسبة المئوية للسيطرة في كل منطقة
    total_touches = hteam_touches + ateam_touches
    hteam_percentage = hteam_touches / total_touches.replace(0, np.nan) * 100
    ateam_percentage = ateam_touches / total_touches.replace(0, np.nan) * 100

    # تحديد مناطق السيطرة
    domination = np.zeros((5, 6), dtype=object)
    for i in range(5):
        for j in range(6):
            h_percent = hteam_percentage.iloc[i, j]
            a_percent = ateam_percentage.iloc[i, j]
            if pd.isna(h_percent) or pd.isna(a_percent):
                domination[i, j] = 'contested'
            elif h_percent > 55:
                domination[i, j] = 'home'
            elif a_percent > 55:
                domination[i, j] = 'away'
            else:
                domination[i, j] = 'contested'

    # ترك فراغ كما طلبت
    # (الفراغ موجود هنا)

    # إعداد تصميم عصري
    # تدرج لوني داكن وعصري (من الأسود إلى الأزرق الداكن)
    gradient = LinearSegmentedColormap.from_list(
        "modern_gradient", ['#0D1B2A', '#1B263B'], N=100)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = Y
    ax.imshow(
        Z,
        extent=[
            0,
            105,
            0,
            68],
        cmap=gradient,
        alpha=0.9,
        aspect='auto',
        zorder=0)

    # رسم المناطق مع تأثيرات ظلال
    bin_width = 105 / 6  # عرض المربع
    bin_height = 68 / 5  # ارتفاع المربع
    for i in range(5):
        for j in range(6):
            x_start = j * bin_width
            y_start = i * bin_height
            if domination[i, j] == 'home':
                color = hcol
                alpha = 0.7
                percentage = hteam_percentage.iloc[i, j]
            elif domination[i, j] == 'away':
                color = acol
                alpha = 0.7
                percentage = ateam_percentage.iloc[i, j]
            else:
                color = '#555555'  # لون رمادي داكن للمناطق المتنازع عليها
                alpha = 0.3
                percentage = None

            # رسم المستطيل مع ظل
            rect = patches.Rectangle(
                (x_start,
                 y_start),
                bin_width,
                bin_height,
                linewidth=1.5,
                edgecolor=line_color,
                facecolor=color,
                alpha=alpha,
                zorder=1)
            rect.set_path_effects([path_effects.withStroke(
                linewidth=3, foreground='black', alpha=0.5)])  # إضافة ظل
            ax.add_patch(rect)

            # إضافة النسبة المئوية في منتصف المستطيل
            if percentage is not None and not pd.isna(percentage):
                text = ax.text(x_start + bin_width / 2,
                               y_start + bin_height / 2,
                               f'{percentage:.1f}%',
                               color='white',
                               fontsize=8,
                               ha='center',
                               va='center',
                               weight='bold',
                               zorder=3)
                text.set_path_effects(
                    [path_effects.withStroke(linewidth=2, foreground='black')])

    # رسم الملعب في الأعلى
    pitch = Pitch(
        pitch_type='uefa',
        line_color=line_color,
        linewidth=2,
        corner_arcs=True)
    pitch.draw(ax=ax)

    # ضبط zorder لخطوط الملعب يدويًا
    for artist in ax.get_children():
        if isinstance(
                artist,
                plt.Line2D):  # التحقق من أن العنصر هو خط (مثل خطوط الملعب)
            artist.set_zorder(2)

    # إضافة أسهم الهجوم بتصميم عصري (أعلى وأسفل)
    # سهم الفريق المضيف (أعلى)
    arrow1 = ax.arrow(
        5,
        72,
        20,
        0,
        head_width=2,
        head_length=2,
        fc=hcol,
        ec='white',
        linewidth=1.5,
        zorder=4)
    arrow1.set_path_effects([path_effects.withStroke(
        linewidth=3, foreground='black', alpha=0.5)])  # ظل للسهم
    text1 = ax.text(
        15,
        76,
        reshape_arabic_text(
            f'اتجاه هجوم {hteamName}'),
        color=hcol,
        fontsize=10,
        ha='center',
        va='center',
        zorder=4)
    text1.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    # سهم الفريق الضيف (أسفل)
    arrow2 = ax.arrow(100, -4, -20, 0, head_width=2, head_length=2,
                      fc=acol, ec='white', linewidth=1.5, zorder=4)
    arrow2.set_path_effects([path_effects.withStroke(
        linewidth=3, foreground='black', alpha=0.5)])
    text2 = ax.text(
        90,
        -8,
        reshape_arabic_text(
            f'اتجاه هجوم {ateamName}'),
        color=acol,
        fontsize=10,
        ha='center',
        va='center',
        zorder=4)
    text2.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    # إضافة النصوص مع تحسين التصميم
    period_text = 'الشوط الأول: 0-45 دقيقة' if phase_tag == 'First Half' else 'الشوط الثاني: 45-90 دقيقة' if phase_tag == 'Second Half' else 'الوقت بالكامل: 0-90 دقيقة'
    period = ax.text(
        52.5,
        82,
        reshape_arabic_text(period_text),
        color='white',
        fontsize=14,
        ha='center',
        va='center',
        weight='bold',
        zorder=4)
    period.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    teams = ax.text(
        52.5,
        78,
        reshape_arabic_text(
            f'{hteamName} | المتنازع عليها | {ateamName}'),
        color='white',
        fontsize=12,
        ha='center',
        va='center',
        zorder=4)
    teams.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    # ملاحظات توضيحية
    note1 = ax.text(
        52.5,
        -12,
        reshape_arabic_text('* المنطقة المسيطر عليها: الفريق لديه أكثر من 55% من اللمسات'),
        color='white',
        fontsize=8,
        ha='center',
        va='center',
        zorder=4)
    note1.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])
    note2 = ax.text(
        52.5,
        -16,
        reshape_arabic_text('* المتنازع عليها: الفريق لديه 45-55% من اللمسات'),
        color='white',
        fontsize=8,
        ha='center',
        va='center',
        zorder=4)
    note2.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])


def analyze_attacking_thirds(df, team_id, team_name, competition_name=None, season_name=None):
    """
    تحليل نسب الهجوم في الأثلاث الثلاثة للثلث الهجومي (يسار، وسط، يمين)
    """
    json_data = st.session_state.json_data

    # تحديد ألوان الفريق
    team_color = hcol if team_id == json_data['home']['teamId'] else acol
    opponent_id = json_data['away']['teamId'] if team_id == json_data['home']['teamId'] else json_data['home']['teamId']
    opponent_name = json_data['away']['name'] if team_id == json_data['home']['teamId'] else json_data['home']['name']
    opponent_color = acol if team_id == json_data['home']['teamId'] else hcol

    # استخراج أحداث الفريق في الثلث الهجومي فقط (x >= 66.7)
    team_events = df[df['teamId'] == team_id]
    final_third_events = team_events[team_events['x'] >= 66.7]

    # تقسيم الثلث الهجومي إلى ثلاثة مناطق عرضية
    left_zone = final_third_events[final_third_events['y'] <= 22.7]
    center_zone = final_third_events[(final_third_events['y'] > 22.7) & (final_third_events['y'] <= 45.3)]
    right_zone = final_third_events[final_third_events['y'] > 45.3]

    # حساب النسب المئوية
    total = len(left_zone) + len(center_zone) + len(right_zone)
    left_pct = (len(left_zone) / total * 100) if total > 0 else 0
    center_pct = (len(center_zone) / total * 100) if total > 0 else 0
    right_pct = (len(right_zone) / total * 100) if total > 0 else 0

    # رسم الملعب
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color, stripe=False, goal_type='box')
    pitch.draw(ax=ax)

    # تحديد اتجاه الهجوم (من اليسار إلى اليمين)
    attack_direction = 'right'  # يمكن تغييره إلى 'left' حسب الحاجة

    # رسم مناطق الثلث الهجومي بتدرج لوني جميل
    if attack_direction == 'right':
        # يسار
        ax.add_patch(patches.Rectangle((70, 0), 30, 22.7, 
                                     facecolor=team_color, alpha=0.7, 
                                     edgecolor='white', linewidth=0.5, zorder=1))
        # وسط
        ax.add_patch(patches.Rectangle((70, 22.7), 30, 22.6, 
                                     facecolor=team_color, alpha=0.8, 
                                     edgecolor='white', linewidth=0.5, zorder=1))
        # يمين
        ax.add_patch(patches.Rectangle((70, 45.3), 30, 22.7, 
                                     facecolor=team_color, alpha=0.7, 
                                     edgecolor='white', linewidth=0.5, zorder=1))
    else:
        # يسار
        ax.add_patch(patches.Rectangle((0, 0), 30, 22.7, 
                                     facecolor=team_color, alpha=0.7, 
                                     edgecolor='white', linewidth=0.5, zorder=1))
        # وسط
        ax.add_patch(patches.Rectangle((0, 22.7), 30, 22.6, 
                                     facecolor=team_color, alpha=0.8, 
                                     edgecolor='white', linewidth=0.5, zorder=1))
        # يمين
        ax.add_patch(patches.Rectangle((0, 45.3), 30, 22.7, 
                                     facecolor=team_color, alpha=0.7, 
                                     edgecolor='white', linewidth=0.5, zorder=1))

    # كتابة النسب داخل كل منطقة بتصميم عصري
    # إضافة دوائر خلف النسب لتحسين الوضوح
    if attack_direction == 'right':
        # يسار
        circle1 = plt.Circle((85, 11.35), 7, color='white', alpha=0.2, zorder=2)
        ax.add_artist(circle1)
        ax.text(85, 11.35, f"{left_pct:.1f}%", color='white', fontsize=16, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        # وسط
        circle2 = plt.Circle((85, 34), 7, color='white', alpha=0.2, zorder=2)
        ax.add_artist(circle2)
        ax.text(85, 34, f"{center_pct:.1f}%", color='white', fontsize=16, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        # يمين
        circle3 = plt.Circle((85, 56.7), 7, color='white', alpha=0.2, zorder=2)
        ax.add_artist(circle3)
        ax.text(85, 56.7, f"{right_pct:.1f}%", color='white', fontsize=16, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    else:
        # يسار
        circle1 = plt.Circle((15, 11.35), 7, color='white', alpha=0.2, zorder=2)
        ax.add_artist(circle1)
        ax.text(15, 11.35, f"{left_pct:.1f}%", color='white', fontsize=16, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        # وسط
        circle2 = plt.Circle((15, 34), 7, color='white', alpha=0.2, zorder=2)
        ax.add_artist(circle2)
        ax.text(15, 34, f"{center_pct:.1f}%", color='white', fontsize=16, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        # يمين
        circle3 = plt.Circle((15, 56.7), 7, color='white', alpha=0.2, zorder=2)
        ax.add_artist(circle3)
        ax.text(15, 56.7, f"{right_pct:.1f}%", color='white', fontsize=16, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    # إضافة أسهم لتوضيح اتجاه الهجوم
    if attack_direction == 'right':
        # سهم كبير في أعلى الملعب
        ax.arrow(30, 10, 40, 0, head_width=5, head_length=5, fc=team_color, ec=team_color, zorder=3, linewidth=2)
        # أسهم صغيرة في مناطق الهجوم
        ax.arrow(75, 11.35, 5, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=4, alpha=0.8)
        ax.arrow(75, 34, 5, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=4, alpha=0.8)
        ax.arrow(75, 56.7, 5, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=4, alpha=0.8)
    else:
        # سهم كبير في أعلى الملعب
        ax.arrow(70, 10, -40, 0, head_width=5, head_length=5, fc=team_color, ec=team_color, zorder=3, linewidth=2)
        # أسهم صغيرة في مناطق الهجوم
        ax.arrow(25, 11.35, -5, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=4, alpha=0.8)
        ax.arrow(25, 34, -5, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=4, alpha=0.8)
        ax.arrow(25, 56.7, -5, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=4, alpha=0.8)

    # إضافة مربعات للفرق مع الشعارات (يمكن استبدالها بشعارات حقيقية)
    # مربع الفريق الأول
    team_box = patches.Rectangle((10, 70), 30, 8, facecolor=team_color, alpha=0.8, 
                               edgecolor='white', linewidth=1, zorder=4)
    ax.add_patch(team_box)
    
    # إضافة دائرة كشعار للفريق (يمكن استبدالها بشعار حقيقي)
    team_logo = plt.Circle((15, 74), 3, color='white', zorder=5)
    ax.add_artist(team_logo)
    
    # اسم الفريق
    ax.text(25, 74, reshape_arabic_text(team_name), color='white', fontsize=12, 
            fontweight='bold', ha='center', va='center', zorder=5,
            path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    
    # مربع الفريق الثاني
    opponent_box = patches.Rectangle((60, 70), 30, 8, facecolor=opponent_color, alpha=0.8, 
                                   edgecolor='white', linewidth=1, zorder=4)
    ax.add_patch(opponent_box)
    
    # إضافة دائرة كشعار للفريق المنافس
    opponent_logo = plt.Circle((65, 74), 3, color='white', zorder=5)
    ax.add_artist(opponent_logo)
    
    # اسم الفريق المنافس
    ax.text(75, 74, reshape_arabic_text(opponent_name), color='white', fontsize=12, 
            fontweight='bold', ha='center', va='center', zorder=5,
            path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])

    # إضافة نص يوضح اتجاه الهجوم
    attack_text = reshape_arabic_text("اتجاه الهجوم")
    if attack_direction == 'right':
        ax.text(50, 10, attack_text, color='white', fontsize=10, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    else:
        ax.text(50, 10, attack_text, color='white', fontsize=10, 
                fontweight='bold', ha='center', va='center', zorder=3,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])

    # عنوان الرسم بتصميم عصري
    title_text = f"{team_name} - {opponent_name}"
    subtitle_text = reshape_arabic_text('تحليل مناطق الهجوم')
    
    # إضافة خلفية للعنوان
    title_bg = patches.Rectangle((0.25, 0.95), 0.5, 0.05, transform=fig.transFigure, 
                               facecolor=team_color, alpha=0.7, edgecolor='white', 
                               linewidth=1, zorder=9)
    fig.patches.append(title_bg)
    
    # إضافة العنوان الرئيسي
    fig.text(0.5, 0.975, title_text, fontsize=20, color='white', ha='center', 
             fontweight='bold', zorder=10,
             path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    
    # إضافة العنوان الفرعي
    fig.text(0.5, 0.94, subtitle_text, fontsize=16, color='white', ha='center', zorder=10,
             path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])

    # إضافة العلامة المائية إذا كانت مفعلة
    if watermark_enabled:
        add_watermark(fig, text=watermark_text, alpha=watermark_opacity, 
                     fontsize=watermark_size, color=watermark_color,
                     x_pos=watermark_x, y_pos=watermark_y, 
                     ha=watermark_ha, va=watermark_va)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig

def calculate_team_ppda(
    events_df: pd.DataFrame,
    team: str,
    region: str = 'opponent_defensive_third',
    custom_threshold: float = None,
    pitch_units: float = 105,
    period: str = None,
    include_pressure: bool = True,
    simulate_pressure: bool = True,
    min_def_actions: int = 5,  # زيادة الحد الأدنى للأفعال الدفاعية
    max_pressure_distance: float = 5.0,
    swap_sides_second_half: bool = True,
    use_extended_defs: bool = False,
    calibration_factor_low_defs: float = 0.7,  # معايرة أقل عدوانية
    min_pass_distance: float = 5.0  # الحد الأدنى لمسافة التمريرة
) -> dict:
    try:
        # نسخ إطار البيانات والتحقق من الإحداثيات
        df = events_df.copy()
        for col in ['x', 'y', 'endX', 'endY']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, pitch_units if col in ['x', 'endX'] else 68)
                if df[col].isna().any():
                    df = df.dropna(subset=[col])

        # تصفية الفترة
        if period:
            df = df[df['period'] == period]
            if df.empty:
                st.warning(f"لا توجد بيانات للفترة {period} لفريق {team}.")
                return {}

        # التعامل مع تبديل الجوانب في الشوط الثاني
        if swap_sides_second_half and not period:
            first_half_x = df[df['period'] == 'FirstHalf']['x'].mean()
            second_half_x = df[df['period'] == 'SecondHalf']['x'].mean()
            if abs(first_half_x - second_half_x) > pitch_units / 3:
                df.loc[df['period'] == 'SecondHalf', 'x'] = pitch_units - df.loc[df['period'] == 'SecondHalf', 'x']
                df.loc[df['period'] == 'SecondHalf', 'y'] = 68 - df.loc[df['period'] == 'SecondHalf', 'y']
                if 'endX' in df.columns and 'endY' in df.columns:
                    df.loc[df['period'] == 'SecondHalf', 'endX'] = pitch_units - df.loc[df['period'] == 'SecondHalf', 'endX']
                    df.loc[df['period'] == 'SecondHalf', 'endY'] = 68 - df.loc[df['period'] == 'SecondHalf', 'endY']

        # تحديد الأفعال الدفاعية (استبعاد Fouls غير مؤثرة)
        defs = ['Tackle', 'Interception', 'BlockedPass', 'Challenge']
        if include_pressure and 'Pressure' in df['type'].unique():
            defs.append('Pressure')
        if use_extended_defs:
            extended_defs = ['ShieldBallOpp', 'BallRecovery']
            defs.extend([d for d in extended_defs if d in df['type'].unique()])

        # محاكاة أحداث الضغط (مع وزن أقل)
        if simulate_pressure and 'Pressure' not in df['type'].unique():
            passes = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
            potential_pressure = df[
                (df['type'].isin(['Tackle', 'Challenge', 'Interception', 'BallRecovery'])) &
                (df['outcomeType'] == 'Successful') &
                (~df['qualifiers'].astype(str).str.contains('Error|Missed', na=False))
            ]
            pressure_count = 0
            if not potential_pressure.empty and not passes.empty:
                pressure_events = []
                for _, pressure_row in potential_pressure.iterrows():
                    relevant_passes = passes[
                        (passes['teamName'] != pressure_row['teamName']) &
                        (pressure_row['cumulative_mins'] >= passes['cumulative_mins']) &
                        (pressure_row['cumulative_mins'] <= passes['cumulative_mins'] + 4/60)
                    ]
                    for _, pass_row in relevant_passes.iterrows():
                        distance = ((pressure_row['x'] - pass_row['x'])**2 + (pressure_row['y'] - pass_row['y'])**2)**0.5
                        if distance <= max_pressure_distance:
                            pressure_event = pressure_row.copy()
                            pressure_event['type'] = 'Pressure'
                            pressure_event['pressure_weight'] = 0.5  # وزن أقل للضغط المحاكى
                            pressure_events.append(pressure_event)
                            pressure_count += 1
                if pressure_events:
                    pressure_df = pd.DataFrame(pressure_events)
                    df = pd.concat([df, pressure_df[df.columns.union(['pressure_weight'])]], ignore_index=True)
                    defs.append('Pressure')
                    st.write(f"أحداث 'Pressure' المحاكاة لـ {team}: {str(pressure_count)}")

        # تحديد المنطقة
        team_id = [k for k, v in st.session_state.teams_dict.items() if v == team][0]
        is_home_team = team_id == min(st.session_state.teams_dict.keys())
        if region == 'opponent_defensive_third':
            x_min = pitch_units * (2 / 3) if is_home_team else 0
            x_max = pitch_units if is_home_team else pitch_units / 3
        elif region == 'attacking_third':
            x_min = pitch_units * (2 / 3)
            x_max = pitch_units
        elif region == 'attacking_half':
            x_min = pitch_units / 2
            x_max = pitch_units
        elif region == 'attacking_60':
            x_min = pitch_units * 0.4
            x_max = pitch_units
        elif region == 'whole':
            x_min = 0
            x_max = pitch_units
        elif region == 'custom':
            if custom_threshold is None:
                raise ValueError("يجب تحديد custom_threshold.")
            x_min = custom_threshold
            x_max = pitch_units
        else:
            raise ValueError(f"المنطقة غير معروفة: {region}")

        # تصفية التمريرات الناجحة (مع استبعاد التمريرات القصيرة)
        opponent = [t for t in df['teamName'].unique() if t != team][0]
        passes_allowed = df[
            (df['type'] == 'Pass') &
            (df['outcomeType'] == 'Successful') &
            (df['teamName'] == opponent) &
            (df['x'] >= x_min) &
            (df['x'] <= x_max) &
            (~df['qualifiers'].astype(str).str.contains('Corner|Freekick|Throwin|GoalKick', na=False))
        ]
        # حساب مسافة التمريرة إذا كانت endX/endY متوفرة
        if 'endX' in df.columns and 'endY' in df.columns:
            passes_allowed = passes_allowed.assign(
                pass_distance=np.sqrt((passes_allowed['endX'] - passes_allowed['x'])**2 + (passes_allowed['endY'] - passes_allowed['y'])**2)
            )
            passes_allowed = passes_allowed[passes_allowed['pass_distance'] >= min_pass_distance]
        num_passes = len(passes_allowed)
        st.write(f"الفريق: {team}, التمريرات الناجحة المسموح بها (x من {str(x_min)} إلى {str(x_max)}): {str(num_passes)}")

        # تصفية الأفعال الدفاعية (مع وزن للضغط المحاكى)
        defensive_actions = df[
            (df['type'].isin(defs)) &
            (df['teamName'] == team) &
            (df['x'] >= x_min) &
            (df['x'] <= x_max) &
            (~df['qualifiers'].astype(str).str.contains('Offensive|Tactical', na=False))
        ]
        # حساب عدد الأفعال الدفاعية مع الأوزان
        if 'pressure_weight' in defensive_actions.columns:
            defensive_actions['weight'] = defensive_actions['pressure_weight'].fillna(1.0)
            num_defs = defensive_actions['weight'].sum()
        else:
            num_defs = len(defensive_actions)
        st.write(f"الفريق: {team}, الأفعال الدفاعية (x من {str(x_min)} إلى {str(x_max)}): {str(round(num_defs, 2))}")

        # حساب PPDA
        ppda = round(num_passes / num_defs, 2) if num_defs >= min_def_actions else None
        pressure_ratio = round((num_defs / num_passes) * 100, 2) if num_passes > 0 and num_defs >= min_def_actions else None

        # التحقق من عدم وجود أفعال دفاعية كافية
        if ppda is None:
            st.warning(f"لا يمكن حساب PPDA لفريق {team}: عدد الأفعال الدفاعية قليل جدًا ({str(round(num_defs, 2))}).")
            return {
                'Region': region,
                'Threshold_x_min': x_min,
                'Threshold_x_max': x_max,
                'Passes Allowed': num_passes,
                'Defensive Actions': round(num_defs, 2),
                'PPDA': None,
                'Pressure Ratio (%)': None,
                'Action Breakdown': {}
            }

        # معايرة PPDA (محسنة)
        calibration_factor = 1.0
        if ppda > 15 or ppda < 5:
            total_defs = df[df['type'].isin(defs) & (df['teamName'] == team)]['weight'].sum() if 'weight' in df.columns else len(df[df['type'].isin(defs) & (df['teamName'] == team)])
            region_def_ratio = num_defs / (total_defs + 1e-10)
            if ppda > 15:
                calibration_factor = calibration_factor_low_defs if num_defs < 10 else 0.9  # معايرة أقل عدوانية
                if region_def_ratio < 0.2:
                    calibration_factor *= 0.85
            elif ppda < 5:
                calibration_factor = 1.2  # زيادة طفيفة للقيم المنخفضة
            ppda = round(ppda * calibration_factor, 2)
            st.write(f"PPDA لفريق {team} معاير ({str(ppda)} بعد التصحيح). معامل المعايرة: {str(calibration_factor)}.")

        # تحذير إذا كان عدد الأفعال الدفاعية قليل
        if num_defs < 10:
            st.write(f"تحذير: عدد الأفعال الدفاعية لفريق {team} قليل ({str(round(num_defs, 2))}).")

        # توزيع الأفعال الدفاعية
        breakdown = {a: round(defensive_actions[defensive_actions['type'] == a]['weight'].sum(), 2) if 'weight' in defensive_actions.columns else int((defensive_actions['type'] == a).sum()) for a in defs}
        st.write(f"توزيع الأفعال الدفاعية لـ {team}: {str(breakdown)}")

        return {
            'Region': region,
            'Threshold_x_min': x_min,
            'Threshold_x_max': x_max,
            'Passes Allowed': num_passes,
            'Defensive Actions': round(num_defs, 2),
            'PPDA': ppda,
            'Pressure Ratio (%)': pressure_ratio,
            'Action Breakdown': breakdown
        }

    except Exception as e:
        st.error(f"خطأ في حساب PPDA لفريق {team}: {str(e)}")
        return {}

# دالة لحساب PPDA لكلا الفريقين
def calculate_ppda_separate(
    events_df: pd.DataFrame,
    region: str = 'opponent_defensive_third',
    custom_threshold: float = None,
    pitch_units: float = 105,
    period: str = None,
    include_pressure: bool = True,
    simulate_pressure: bool = True,
    min_def_actions: int = 5,
    max_pressure_distance: float = 5.0,
    swap_sides_second_half: bool = True,
    use_extended_defs: bool = False
) -> dict:
    try:
        teams = events_df['teamName'].unique()
        if len(teams) != 2:
            raise ValueError(f"يتوقع وجود فريقين، تم العثور على: {teams}")

        results = {}
        for team in teams:
            team_max_pressure_distance = team_params.get(team, {}).get('max_pressure_distance', max_pressure_distance)
            team_calibration_factor = team_params.get(team, {}).get('calibration_factor', 0.7)
            st.write(f"حساب PPDA لـ {team} باستخدام max_pressure_distance={str(team_max_pressure_distance)}, calibration_factor_low_defs={str(team_calibration_factor)}")
            results[team] = calculate_team_ppda(
                events_df,
                team,
                region,
                custom_threshold,
                pitch_units,
                period,
                include_pressure,
                simulate_pressure,
                min_def_actions,
                team_max_pressure_distance,
                swap_sides_second_half,
                use_extended_defs,
                team_calibration_factor,
                min_pass_distance=5.0
            )
        return results
    except Exception as e:
        st.error(f"خطأ في حساب PPDA: {str(e)}")
        return {}
def plot_match_stats(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    """
    رسم إحصائيات المباراة بين فريقين
    
    المعلمات:
    ax : matplotlib.axes.Axes
        محور الرسم
    df : pandas.DataFrame
        إطار بيانات أحداث المباراة
    hteamName : str
        اسم الفريق المضيف
    ateamName : str
        اسم الفريق الضيف
    hcol : str
        لون الفريق المضيف
    acol : str
        لون الفريق الضيف
    bg_color : str
        لون الخلفية
    line_color : str
        لون الخطوط
    
    Returns:
    -------
    pandas.DataFrame
        إطار بيانات يحتوي على جميع الإحصائيات المحسوبة
    """
    # إنشاء إطار بيانات لتخزين الإحصائيات
    stats_df = pd.DataFrame()
    
    # إحصائيات التمرير
    # الاستحواذ %
    hpossdf = df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & (df['type'] == 'Pass')]
    apossdf = df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & (df['type'] == 'Pass')]
    
    total_passes = len(hpossdf) + len(apossdf)
    if total_passes > 0:
        hposs = round((len(hpossdf) / total_passes) * 100, 1)
        aposs = round((len(apossdf) / total_passes) * 100, 1)
    else:
        hposs = 0
        aposs = 0
    
    # Field Tilt % (نسبة اللعب في الثلث الهجومي)
    hftdf = df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & (df['isTouch'] == True) & (df['x'] >= 70)]
    aftdf = df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & (df['isTouch'] == True) & (df['x'] >= 70)]
    
    total_final_third = len(hftdf) + len(aftdf)
    if total_final_third > 0:
        hft = round((len(hftdf) / total_final_third) * 100, 1)
        aft = round((len(aftdf) / total_final_third) * 100, 1)
    else:
        hft = 0
        aft = 0
    
    # إجمالي التمريرات
    htotalPass = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & (df['type'] == 'Pass')])
    atotalPass = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & (df['type'] == 'Pass')])
    
    # التمريرات الناجحة
    hAccPass = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                      (df['type'] == 'Pass') & 
                      (df['outcomeType'] == 'Successful')])
    aAccPass = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                      (df['type'] == 'Pass') & 
                      (df['outcomeType'] == 'Successful')])
    
    # نسبة دقة التمريرات
    hPassAcc = round((hAccPass / htotalPass * 100), 1) if htotalPass > 0 else 0
    aPassAcc = round((aAccPass / atotalPass * 100), 1) if atotalPass > 0 else 0
    
    # التمريرات الناجحة (بدون الثلث الدفاعي)
    hAccPasswdt = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                         (df['type'] == 'Pass') & 
                         (df['outcomeType'] == 'Successful') & 
                         (df['endX'] > 35)])
    aAccPasswdt = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                         (df['type'] == 'Pass') & 
                         (df['outcomeType'] == 'Successful') & 
                         (df['endX'] > 35)])
    
    # الكرات الطويلة
    # استخدام دالة مساعدة للتحقق من وجود نوع معين في qualifiers
    def has_qualifier(qualifiers, qualifier_name, exclude_list=None):
        if isinstance(qualifiers, str):
            # إذا كان qualifiers نصًا، نحاول تحويله إلى قاموس
            try:
                import ast
                qualifiers = ast.literal_eval(qualifiers)
            except:
                return False
        
        if isinstance(qualifiers, dict) and 'type' in qualifiers:
            # إذا كان qualifiers قاموسًا مباشرًا
            if 'displayName' in qualifiers['type'] and qualifiers['type']['displayName'] == qualifier_name:
                if exclude_list:
                    for exclude in exclude_list:
                        if exclude in qualifiers['type']['displayName']:
                            return False
                return True
        elif isinstance(qualifiers, list):
            # إذا كان qualifiers قائمة من القواميس
            for q in qualifiers:
                if isinstance(q, dict) and 'type' in q:
                    if 'displayName' in q['type'] and q['type']['displayName'] == qualifier_name:
                        if exclude_list:
                            for exclude in exclude_list:
                                if exclude in q['type']['displayName']:
                                    return False
                        return True
        
        return False
    
    # تطبيق الدالة المساعدة على البيانات
    df['has_longball'] = df['qualifiers'].apply(lambda x: has_qualifier(x, 'Longball', ['Corner', 'Cross']))
    df['has_cross'] = df['qualifiers'].apply(lambda x: has_qualifier(x, 'Cross'))
    df['has_freekick'] = df['qualifiers'].apply(lambda x: has_qualifier(x, 'Freekick'))
    df['has_corner'] = df['qualifiers'].apply(lambda x: has_qualifier(x, 'Corner'))
    df['has_throwin'] = df['qualifiers'].apply(lambda x: has_qualifier(x, 'ThrowIn'))
    df['has_goalkick'] = df['qualifiers'].apply(lambda x: has_qualifier(x, 'GoalKick'))
    
    # الكرات الطويلة
    hLongB = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                    (df['type'] == 'Pass') & 
                    (df['has_longball'] == True)])
    aLongB = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                    (df['type'] == 'Pass') & 
                    (df['has_longball'] == True)])
    
    # الكرات الطويلة الناجحة
    hAccLongB = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                       (df['type'] == 'Pass') & 
                       (df['has_longball'] == True) & 
                       (df['outcomeType'] == 'Successful')])
    aAccLongB = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                       (df['type'] == 'Pass') & 
                       (df['has_longball'] == True) & 
                       (df['outcomeType'] == 'Successful')])
    
    # نسبة دقة الكرات الطويلة
    hLongBAcc = round((hAccLongB / hLongB * 100), 1) if hLongB > 0 else 0
    aLongBAcc = round((aAccLongB / aLongB * 100), 1) if aLongB > 0 else 0
    
    # العرضيات
    hCrss = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                   (df['type'] == 'Pass') & 
                   (df['has_cross'] == True)])
    aCrss = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                   (df['type'] == 'Pass') & 
                   (df['has_cross'] == True)])
    
    # العرضيات الناجحة
    hAccCrss = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                      (df['type'] == 'Pass') & 
                      (df['has_cross'] == True) & 
                      (df['outcomeType'] == 'Successful')])
    aAccCrss = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                      (df['type'] == 'Pass') & 
                      (df['has_cross'] == True) & 
                      (df['outcomeType'] == 'Successful')])
    
    # نسبة دقة العرضيات
    hCrssAcc = round((hAccCrss / hCrss * 100), 1) if hCrss > 0 else 0
    aCrssAcc = round((aAccCrss / aCrss * 100), 1) if aCrss > 0 else 0
    
    # الركلات الحرة
    hfk = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                (df['type'] == 'Pass') & 
                (df['has_freekick'] == True)])
    afk = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                (df['type'] == 'Pass') & 
                (df['has_freekick'] == True)])
    
    # الركنيات
    hCor = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                 (df['type'] == 'Pass') & 
                 (df['has_corner'] == True)])
    aCor = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                 (df['type'] == 'Pass') & 
                 (df['has_corner'] == True)])
    
    # رميات التماس
    htins = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                  (df['type'] == 'Pass') & 
                  (df['has_throwin'] == True)])
    atins = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                  (df['type'] == 'Pass') & 
                  (df['has_throwin'] == True)])
    
    # ركلات المرمى
    hglkk = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                  (df['type'] == 'Pass') & 
                  (df['has_goalkick'] == True)])
    aglkk = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                  (df['type'] == 'Pass') & 
                  (df['has_goalkick'] == True)])
    
    # المراوغات
    htotalDrb = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                       (df['type'] == 'TakeOn')])
    atotalDrb = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                       (df['type'] == 'TakeOn')])
    
    # المراوغات الناجحة
    hAccDrb = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                     (df['type'] == 'TakeOn') & 
                     (df['outcomeType'] == 'Successful')])
    aAccDrb = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                     (df['type'] == 'TakeOn') & 
                     (df['outcomeType'] == 'Successful')])
    
    # نسبة نجاح المراوغات
    hDrbAcc = round((hAccDrb / htotalDrb * 100), 1) if htotalDrb > 0 else 0
    aDrbAcc = round((aAccDrb / atotalDrb * 100), 1) if atotalDrb > 0 else 0
    
    # التسديدات
    hShots = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                    (df['type'] == 'Shot')])
    aShots = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                    (df['type'] == 'Shot')])
    
    # التسديدات على المرمى
    hShotsOnTarget = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                           (df['type'] == 'Shot') & 
                           (df['outcomeType'] == 'Successful')])
    aShotsOnTarget = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                           (df['type'] == 'Shot') & 
                           (df['outcomeType'] == 'Successful')])
    
    # نسبة التسديدات على المرمى
    hShotsOnTargetPct = round((hShotsOnTarget / hShots * 100), 1) if hShots > 0 else 0
    aShotsOnTargetPct = round((aShotsOnTarget / aShots * 100), 1) if aShots > 0 else 0
    
    # الأهداف
    hGoals = len(df[(df['teamId'] == st.session_state.json_data['home']['teamId']) & 
                    (df['type'] == 'Shot') & 
                    (df['isGoal'] == True)])
    aGoals = len(df[(df['teamId'] == st.session_state.json_data['away']['teamId']) & 
                    (df['type'] == 'Shot') & 
                    (df['isGoal'] == True)])
    
    # تخزين الإحصائيات في إطار البيانات
    stats_df.loc['الاستحواذ %', hteamName] = hposs
    stats_df.loc['الاستحواذ %', ateamName] = aposs
    
    stats_df.loc['نسبة اللعب في الثلث الهجومي %', hteamName] = hft
    stats_df.loc['نسبة اللعب في الثلث الهجومي %', ateamName] = aft
    
    stats_df.loc['إجمالي التمريرات', hteamName] = htotalPass
    stats_df.loc['إجمالي التمريرات', ateamName] = atotalPass
    
    stats_df.loc['التمريرات الناجحة', hteamName] = hAccPass
    stats_df.loc['التمريرات الناجحة', ateamName] = aAccPass
    
    stats_df.loc['دقة التمريرات %', hteamName] = hPassAcc
    stats_df.loc['دقة التمريرات %', ateamName] = aPassAcc
    
    stats_df.loc['الكرات الطويلة', hteamName] = hLongB
    stats_df.loc['الكرات الطويلة', ateamName] = aLongB
    
    stats_df.loc['الكرات الطويلة الناجحة', hteamName] = hAccLongB
    stats_df.loc['الكرات الطويلة الناجحة', ateamName] = aAccLongB
    
    stats_df.loc['دقة الكرات الطويلة %', hteamName] = hLongBAcc
    stats_df.loc['دقة الكرات الطويلة %', ateamName] = aLongBAcc
    
    stats_df.loc['العرضيات', hteamName] = hCrss
    stats_df.loc['العرضيات', ateamName] = aCrss
    
    stats_df.loc['العرضيات الناجحة', hteamName] = hAccCrss
    stats_df.loc['العرضيات الناجحة', ateamName] = aAccCrss
    
    stats_df.loc['دقة العرضيات %', hteamName] = hCrssAcc
    stats_df.loc['دقة العرضيات %', ateamName] = aCrssAcc
    
    stats_df.loc['الركلات الحرة', hteamName] = hfk
    stats_df.loc['الركلات الحرة', ateamName] = afk
    
    stats_df.loc['الركنيات', hteamName] = hCor
    stats_df.loc['الركنيات', ateamName] = aCor
    
    stats_df.loc['رميات التماس', hteamName] = htins
    stats_df.loc['رميات التماس', ateamName] = atins
    
    stats_df.loc['ركلات المرمى', hteamName] = hglkk
    stats_df.loc['ركلات المرمى', ateamName] = aglkk
    
    stats_df.loc['المراوغات', hteamName] = htotalDrb
    stats_df.loc['المراوغات', ateamName] = atotalDrb
    
    stats_df.loc['المراوغات الناجحة', hteamName] = hAccDrb
    stats_df.loc['المراوغات الناجحة', ateamName] = aAccDrb
    
    stats_df.loc['نسبة نجاح المراوغات %', hteamName] = hDrbAcc
    stats_df.loc['نسبة نجاح المراوغات %', ateamName] = aDrbAcc
    
    stats_df.loc['التسديدات', hteamName] = hShots
    stats_df.loc['التسديدات', ateamName] = aShots
    
    stats_df.loc['التسديدات على المرمى', hteamName] = hShotsOnTarget
    stats_df.loc['التسديدات على المرمى', ateamName] = aShotsOnTarget
    
    stats_df.loc['نسبة التسديدات على المرمى %', hteamName] = hShotsOnTargetPct
    stats_df.loc['نسبة التسديدات على المرمى %', ateamName] = aShotsOnTargetPct
    
    stats_df.loc['الأهداف', hteamName] = hGoals
    stats_df.loc['الأهداف', ateamName] = aGoals
    
    # تحديد الإحصائيات التي سيتم عرضها في الرسم البياني
    display_stats = [
        'الاستحواذ %',
        'نسبة اللعب في الثلث الهجومي %',
        'إجمالي التمريرات',
        'دقة التمريرات %',
        'الكرات الطويلة',
        'دقة الكرات الطويلة %',
        'العرضيات',
        'دقة العرضيات %',
        'الركلات الحرة',
        'الركنيات',
        'المراوغات',
        'نسبة نجاح المراوغات %',
        'التسديدات',
        'التسديدات على المرمى',
        'الأهداف'
    ]
    
    # إنشاء إطار بيانات للعرض
    display_df = stats_df.loc[display_stats]
    
    # تعيين الألوان للفرق
    team_colors = {hteamName: hcol, ateamName: acol}
    
    # إعداد الرسم البياني
    ax.set_facecolor(bg_color)
    
    # عدد الإحصائيات المعروضة
    n_stats = len(display_stats)
    
    # تحديد مواضع الأشرطة
    y_pos = np.arange(n_stats)
    bar_height = 0.35
    
    # رسم الأشرطة للفريق المضيف
    home_bars = ax.barh(y_pos - bar_height/2, display_df[hteamName], bar_height, 
                        color=hcol, alpha=0.7, label=hteamName)
    
    # رسم الأشرطة للفريق الضيف
    away_bars = ax.barh(y_pos + bar_height/2, display_df[ateamName], bar_height, 
                        color=acol, alpha=0.7, label=ateamName)
    
    # إضافة قيم الإحصائيات على الأشرطة
    for i, bar in enumerate(home_bars):
        stat_value = display_df[hteamName].iloc[i]
        if isinstance(stat_value, (int, float)):
            if '%' in display_stats[i]:  # إذا كانت نسبة مئوية
                text = f"{stat_value:.1f}%"
            else:  # إذا كانت قيمة عددية
                text = f"{int(stat_value)}"
            
            # تحديد موضع النص (داخل أو خارج الشريط)
            width = bar.get_width()
            if width > max(display_df[hteamName].max(), display_df[ateamName].max()) * 0.15:
                # داخل الشريط
                ax.text(width / 2, bar.get_y() + bar.get_height() / 2, text,
                        ha='center', va='center', color='white', fontweight='bold',
                        path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
            else:
                # خارج الشريط
                ax.text(width + max(display_df[hteamName].max(), display_df[ateamName].max()) * 0.02, 
                        bar.get_y() + bar.get_height() / 2, text,
                        ha='left', va='center', color='white', fontweight='bold',
                        path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
    
    for i, bar in enumerate(away_bars):
        stat_value = display_df[ateamName].iloc[i]
        if isinstance(stat_value, (int, float)):
            if '%' in display_stats[i]:  # إذا كانت نسبة مئوية
                text = f"{stat_value:.1f}%"
            else:  # إذا كانت قيمة عددية
                text = f"{int(stat_value)}"
            
            # تحديد موضع النص (داخل أو خارج الشريط)
            width = bar.get_width()
            if width > max(display_df[hteamName].max(), display_df[ateamName].max()) * 0.15:
                # داخل الشريط
                ax.text(width / 2, bar.get_y() + bar.get_height() / 2, text,
                        ha='center', va='center', color='white', fontweight='bold',
                        path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
            else:
                # خارج الشريط
                ax.text(width + max(display_df[hteamName].max(), display_df[ateamName].max()) * 0.02, 
                        bar.get_y() + bar.get_height() / 2, text,
                        ha='left', va='center', color='white', fontweight='bold',
                        path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
    
    # تعيين أسماء الإحصائيات
    ax.set_yticks(y_pos)
    ax.set_yticklabels([reshape_arabic_text(stat) for stat in display_stats], fontsize=12)
    
    # تعيين حدود المحور الأفقي
    max_value = max(display_df[hteamName].max(), display_df[ateamName].max())
    ax.set_xlim(0, max_value * 1.2)
    
    # إزالة الإطار والشبكة
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color(line_color)
    
    # إزالة علامات المحور الأفقي
    ax.set_xticks([])
    
    # إضافة وسيلة إيضاح
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, 
              fontsize=12, labelcolor='white')
    
    # إضافة عنوان
    ax.set_title(reshape_arabic_text('إحصائيات المباراة'), color='white', fontsize=16, 
                 fontweight='bold', pad=20,
                 path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    
    return stats_df

# واجهة Streamlit
st.title("تحليل مباراة كرة القدم")
uploaded_html = st.file_uploader("قم برفع ملف HTML للمباراة:", type=["html"])
uploaded_json = st.file_uploader("أو قم بتحميل ملف JSON (اختياري):", type="json")

if st.button("تحليل المباراة"):
    with st.spinner("جارٍ استخراج بيانات المباراة..."):
        st.session_state.json_data = None
        st.session_state.df = pd.DataFrame({
            'period': ['FirstHalf', 'FirstHalf', 'SecondHalf', 'SecondHalf', 'FirstHalf', 'SecondHalf'],
            'teamName': ['TeamA', 'TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamA'],
            'type': ['Pass', 'Pass', 'Pass', 'Goal', 'Pass', 'Carry'],
            'outcomeType': ['Successful', 'Successful', 'Successful', 'Successful', 'Successful', 'Successful'],
            'name': ['Player1', 'Player2', 'Player1', 'Player3', 'Player4', 'Player2'],
            'x': [10, 20, 30, 90, 50, 40],
            'y': [10, 20, 30, 34, 40, 50],
            'endX': [20, 30, 40, np.nan, 60, 50],
            'endY': [20, 30, 40, np.nan, 50, 60],
            'qualifiers': ['', '', '', 'Goal', '', ''],
            'shirtNo': [1, 2, 1, 3, 4, 2],
            'position': ['DC', 'DC', 'DC', 'FW', 'MC', 'DC'],
            'isFirstEleven': [True, True, True, True, True, True],
            'playerId': [101, 102, 101, 103, 104, 102],
            'isTouch': [True, True, True, True, True, True],
            'cumulative_mins': [5, 10, 50, 60, 20, 70]
        })
        st.session_state.players_df = pd.DataFrame({
            'name': ['Player1', 'Player2', 'Player3', 'Player4'],
            'shirtNo': [1, 2, 3, 4],
            'position': ['DC', 'DC', 'FW', 'MC'],
            'isFirstEleven': [True, True, True, True],
            'playerId': [101, 102, 103, 104]
        })
        st.session_state.teams_dict = {1: 'TeamA', 2: 'TeamB'}
        st.session_state.analysis_triggered = True
        st.success("تم تحميل البيانات التجريبية بنجاح!")
        
        if uploaded_json:
            try:
                st.session_state.json_data = json.load(uploaded_json)
            except Exception as e:
                st.error(f"خطأ في تحميل ملف JSON: {str(e)}")
        elif uploaded_html:
            try:
                st.session_state.json_data = extract_match_dict_from_html(uploaded_html)
            except Exception as e:
                st.error(f"خطأ في معالجة ملف HTML: {str(e)}")
        else:
            st.error("يرجى رفع ملف HTML أو JSON للمباراة.")

        if st.session_state.json_data:
            st.session_state.df, st.session_state.teams_dict, st.session_state.players_df = get_event_data(
                st.session_state.json_data)
            st.session_state.analysis_triggered = True
            if st.session_state.df is not None and st.session_state.teams_dict and st.session_state.players_df is not None:
                st.success("تم استخراج البيانات بنجاح!")
            else:
                st.error("فشل في معالجة البيانات.")
        else:
            st.error("فشل في جلب بيانات المباراة.")

# عرض التحليل فقط إذا تم استخراج البيانات
# عرض التحليل فقط إذا تم استخراج البيانات
if st.session_state.analysis_triggered and not st.session_state.df.empty and st.session_state.teams_dict and not st.session_state.players_df.empty:
    hteamID = list(st.session_state.teams_dict.keys())[0]
    ateamID = list(st.session_state.teams_dict.keys())[1]
    hteamName = st.session_state.teams_dict[hteamID]
    ateamName = st.session_state.teams_dict[ateamID]

    homedf = st.session_state.df[(st.session_state.df['teamName'] == hteamName)]
    awaydf = st.session_state.df[(st.session_state.df['teamName'] == ateamName)]
    hxT = homedf['xT'].sum().round(2) if 'xT' in homedf.columns else 0
    axT = awaydf['xT'].sum().round(2) if 'xT' in awaydf.columns else 0

    hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & 
                             (homedf['type'] == 'Goal') & 
                             (~homedf['qualifiers'].str.contains('OwnGoal', na=False))])
    agoal_count = len(awaydf[(awaydf['teamName'] == ateamName) & 
                             (awaydf['type'] == 'Goal') & 
                             (~awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
    hgoal_count += len(awaydf[(awaydf['teamName'] == ateamName) & 
                              (awaydf['type'] == 'Goal') & 
                              (awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
    agoal_count += len(homedf[(homedf['teamName'] == hteamName) & 
                              (homedf['type'] == 'Goal') & 
                              (homedf['qualifiers'].str.contains('OwnGoal', na=False))])

    hftmb_tid = fotmob_team_ids.get(hteamName, 0)
    aftmb_tid = fotmob_team_ids.get(ateamName, 0)

    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')

    # تعريف علامات التبويب داخل try-except
    try:
        tab1, tab2, tab3, tab4 = st.tabs(['تحليل الفريق', 'تحليل اللاعبين', 'إحصائيات المباراة', 'أفضل اللاعبين'])
    except Exception as e:
        st.error(f"خطأ في إنشاء التبويبات: {str(e)}")
        st.stop()

    # علامات التبويب

        tab1, tab2, tab3, tab4 = st.tabs(
            ['تحليل الفريق', 'تحليل اللاعبين', 'إحصائيات المباراة', 'أفضل اللاعبين'])
    except Exception as e:
        st.error(f"خطأ في إنشاء التبويبات: {str(e)}")
        st.stop()
with tab1:
    an_tp = st.selectbox('نوع التحليل:', [
        'شبكة التمريرات',
        'مناطق الهجوم',
        'Defensive Actions Heatmap',
        'Progressive Passes',
        'Progressive Carries',
        'Shotmap',
        'إحصائيات الحراس',
        'Match Momentum',
        reshape_arabic_text('Zone14 & Half-Space Passes'),
        reshape_arabic_text('Final Third Entries'),
        reshape_arabic_text('Box Entries'),
        reshape_arabic_text('High-Turnovers'),
        reshape_arabic_text('Chances Creating Zones'),
        reshape_arabic_text('Crosses'),
        reshape_arabic_text('Team Domination Zones'),
        reshape_arabic_text('Pass Target Zones'),
        'Attacking Thirds',
        reshape_arabic_text('PPDA')
    ], index=0, key='analysis_type')

    if an_tp == 'شبكة التمريرات':
        st.subheader('شبكة التمريرات')
        team_choice = st.selectbox('اختر الفريق:', [hteamName, ateamName], key='team_choice')
        phase_tag = st.selectbox('اختر الفترة:', ['Full Time', 'First Half', 'Second Half'], key='phase_tag')
        
        # إنشاء الرسم
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color, dpi=150)
        
        # استدعاء pass_network
        try:
            pass_btn = pass_network(
                ax,
                team_choice,
                hcol if team_choice == hteamName else acol,
                phase_tag,
                hteamName,
                ateamName,
                hgoal_count,
                agoal_count,
                hteamID,
                ateamID
            )
            # إضافة العلامة المائية إذا كانت مفعلة
            if watermark_enabled:
                fig = add_watermark(fig, text=watermark_text, alpha=watermark_opacity, 
                       fontsize=watermark_size, color=watermark_color,
                       x_pos=watermark_x, y_pos=watermark_y, 
                       ha=watermark_ha, va=watermark_va)
            
            st.pyplot(fig)
            if pass_btn is not None and not pass_btn.empty:
                st.dataframe(pass_btn, hide_index=True)
            else:
                st.warning("لا توجد بيانات تمريرات للعرض.")
        except Exception as e:
            st.error(f"خطأ في إنشاء شبكة التمريرات: {str(e)}")

    elif an_tp == 'مناطق الهجوم':
        st.subheader("تحليل مناطق الهجوم")
    
    # اختيار الفريق مع إضافة key فريد
    team_ids = list(st.session_state.teams_dict.keys())
    team_names = list(st.session_state.teams_dict.values())
    selected_team_index = st.selectbox("اختر الفريق:", range(len(team_ids)),
                                      format_func=lambda x: team_names[x],
                                      key="attacking_thirds_team_select")  # إضافة key فريد
    selected_team_id = team_ids[selected_team_index]
    selected_team_name = team_names[selected_team_index]
    
    # اختيار المسابقة والموسم (اختياري)
    competition_name = st.text_input("اسم المسابقة (اختياري):", "", key="attacking_thirds_competition")  # إضافة key فريد
    
    # تحليل وعرض مناطق الهجوم مباشرة بدون زر
    with st.spinner("جاري تحليل مناطق الهجوم..."):
        fig = analyze_attacking_thirds(st.session_state.df, selected_team_id,
                                      selected_team_name, competition_name)
        st.pyplot(fig)
        
        # حفظ الصورة
        save_path = f"{selected_team_name}_attacking_thirds.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        st.success(f"تم حفظ الصورة بنجاح: {save_path}")

    if an_tp == reshape_arabic_text('Team Domination Zones'):
        st.subheader(reshape_arabic_text('مناطق سيطرة الفريق'))
        phase_tag = st.selectbox(
            'اختر الفترة:', ['Full Time', 'First Half', 'Second Half'], key='phase_tag_domination')
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
        team_domination_zones(
            ax,
            phase_tag,
            hteamName,
            ateamName,
            hcol,
            acol,
            bg_color,
            line_color,
            gradient_colors)
        # إضافة عنوان أعلى الرسم
        fig.text(
            0.5, 0.98,
            reshape_arabic_text(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}'),
            fontsize=16, fontweight='bold', ha='center', va='center', color='white')
        fig.text(0.5, 0.94, reshape_arabic_text('مناطق السيطرة'),
                 fontsize=14, ha='center', va='center', color='white')
        # إضافة العلامة المائية إذا كانت مفعلة
        if watermark_enabled:
            fig = add_watermark(fig, text=watermark_text, alpha=watermark_opacity, 
                       fontsize=watermark_size, color=watermark_color,
                       x_pos=watermark_x, y_pos=watermark_y, 
                       ha=watermark_ha, va=watermark_va)
        
        st.pyplot(fig)

    elif an_tp == reshape_arabic_text('PPDA'):
        st.subheader(reshape_arabic_text('معدل الضغط (PPDA)'))
        st.write(reshape_arabic_text("PPDA: عدد التمريرات الناجحة التي يسمح بها الفريق مقابل كل فعل دفاعي في الثلث الدفاعي للخصم. القيمة الأقل تشير إلى ضغط دفاعي أقوى (عادة 5-15)."))

    # إضافة خيارات لتخصيص PPDA
    period_choice = st.selectbox(
        reshape_arabic_text('اختر الفترة:'),
        ['Full Match', 'First Half', 'Second Half'],
        key='ppda_period'
    )
    period_map = {
        'Full Match': None,
        'First Half': 'FirstHalf',
        'Second Half': 'SecondHalf'
    }
    selected_period = period_map.get(period_choice, None)

    region_choice = st.selectbox(
        reshape_arabic_text('اختر المنطقة:'),
        ['الثلث الدفاعي للخصم', 'الثلث الهجومي', 'نصف الملعب الهجومي', '60% من الملعب الهجومي', 'الملعب بأكمله'],
        index=0,
        key='ppda_region'
    )
    region_map = {
        'الثلث الدفاعي للخصم': 'opponent_defensive_third',
        'الثلث الهجومي': 'attacking_third',
        'نصف الملعب الهجومي': 'attacking_half',
        '60% من الملعب الهجومي': 'attacking_60',
        'الملعب بأكمله': 'whole'
    }
    selected_region = region_map.get(region_choice, 'opponent_defensive_third')

    simulate_pressure = st.checkbox(
        reshape_arabic_text('محاكاة أحداث الضغط (إذا لم تكن متوفرة)'),
        value=True,
        key='simulate_pressure'
    )

    st.subheader(reshape_arabic_text('إعدادات مخصصة لكل فريق'))
    teams = st.session_state.df['teamName'].unique()
    team_params = {}
    for team in teams:
        with st.expander(reshape_arabic_text(f'إعدادات {team}')):
            max_pressure_distance = st.slider(
                reshape_arabic_text(f'الحد الأقصى للمسافة لمحاكاة الضغط لـ {team} (بالأمتار):'),
                min_value=3.0,
                max_value=10.0,
                value=6.0 if team == 'Celta Vigo' else 5.0,
                step=1.0,
                key=f'max_pressure_distance_{team}'
            )
            calibration_factor = st.slider(
                reshape_arabic_text(f'معامل المعايرة للأفعال القليلة لـ {team}:'),
                min_value=0.3,
                max_value=1.0,
                value=0.4 if team == 'Celta Vigo' else 0.5,
                step=0.1,
                key=f'calibration_factor_{team}'
            )
            team_params[team] = {
                'max_pressure_distance': max_pressure_distance,
                'calibration_factor': calibration_factor
            }

    swap_sides = st.checkbox(
        reshape_arabic_text('تبديل الجوانب في الشوط الثاني'),
        value=True,
        key='swap_sides'
    )

    use_extended_defs = st.checkbox(
        reshape_arabic_text('استخدام أفعال دفاعية موسعة (مثل ShieldBallOpp)'),
        value=False,
        key='use_extended_defs'
    )

    try:
        results = {}
        for team in teams:
            st.write(f"حساب PPDA لـ {team}")
            results[team] = calculate_team_ppda(
                st.session_state.df,
                team,
                region=selected_region,
                period=selected_period,
                min_def_actions=1,
                include_pressure=True,
                simulate_pressure=simulate_pressure,
                max_pressure_distance=team_params[team]['max_pressure_distance'],
                swap_sides_second_half=swap_sides,
                use_extended_defs=use_extended_defs,
                calibration_factor_low_defs=team_params[team]['calibration_factor']
            )

        if not results:
            st.error("لا توجد نتائج PPDA متاحة.")
        else:
            ppda_df = pd.DataFrame.from_dict(results, orient='index')

            st.subheader(reshape_arabic_text('نتائج PPDA'))
            st.dataframe(
                ppda_df[['Passes Allowed', 'Defensive Actions', 'PPDA', 'Pressure Ratio (%)']],
                use_container_width=True
            )

            st.subheader(reshape_arabic_text('تفاصيل الأفعال الدفاعية'))
            for team, result in results.items():
                st.write(reshape_arabic_text(f"الفريق: {team}"))
                action_df = pd.DataFrame.from_dict(result['Action Breakdown'], orient='index', columns=['Count'])
                st.dataframe(action_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
            ax.set_facecolor('#1a1a1a')
            colors = sns.color_palette("husl", len(ppda_df))
            bars = ax.bar(ppda_df.index, ppda_df['PPDA'].fillna(0), color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)

            for bar in bars:
                height = bar.get_height()
                label = f'{height:.2f}' if height > 0 else 'غير متاح'
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 0.5,
                    label, ha='center', va='bottom', color='white',
                    fontsize=12, fontweight='bold',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]
                )

            ax.set_title(
                reshape_arabic_text(f'معدل الضغط (PPDA) لكل فريق - {period_choice}'),
                fontsize=16, color='white', pad=20, fontweight='bold'
            )
            ax.set_xlabel(reshape_arabic_text('الفريق'), fontsize=12, color='white')
            ax.set_ylabel('PPDA', fontsize=12, color='white')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.tick_params(colors='white', labelsize=10)

            ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='white')
            ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
            ax.text(0, 10.5, reshape_arabic_text('متوسط PPDA في الدوري'), color='white')

            plt.tight_layout()
            # إضافة العلامة المائية إذا كانت مفعلة
            if watermark_enabled:
                fig = add_watermark(fig, text=watermark_text, alpha=watermark_opacity, 
                       fontsize=watermark_size, color=watermark_color,
                       x_pos=watermark_x, y_pos=watermark_y, 
                       ha=watermark_ha, va=watermark_va)
        
    except Exception as e:
        st.error(f"حدث خطأ: {e}")

with tab3:
    st.subheader(reshape_arabic_text("إحصائيات المباراة"))
    try:
        df = st.session_state.df

        # التأكد من وجود الأعمدة المطلوبة
        required_columns = ['teamId', 'type', 'qualifiers', 'outcomeType', 'x', 'y', 'endX', 'endY', 'isTouch', 'playerId']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"الأعمدة المطلوبة مفقودة: {missing_columns}")
            st.stop()

        # إنشاء الأعمدة بناءً على عمود qualifiers
        df['has_longball'] = df['qualifiers'].str.contains('Longball', na=False) & (~df['qualifiers'].str.contains('Corner', na=False)) & (~df['qualifiers'].str.contains('Cross', na=False))
        df['has_cross'] = df['qualifiers'].str.contains('Cross', na=False)
        df['has_freekick'] = df['qualifiers'].str.contains('Freekick', na=False)
        df['has_corner'] = df['qualifiers'].str.contains('Corner', na=False)
        df['has_throwin'] = df['qualifiers'].str.contains('ThrowIn', na=False)
        df['has_goalkick'] = df['qualifiers'].str.contains('GoalKick', na=False)

        # استخراج معرفات الفرق وتحويلها إلى أعداد صحيحة
        team_ids = [int(tid) for tid in df['teamId'].unique()]
        if not team_ids or len(team_ids) < 2:
            raise ValueError("لا توجد فرق كافية في البيانات لعرض الإحصائيات.")

        # التحقق من وجود team_ids في teams_dict
        missing_teams = [tid for tid in team_ids if tid not in st.session_state.teams_dict]
        if missing_teams:
            raise KeyError(f"معرفات الفرق التالية غير موجودة في teams_dict: {missing_teams}")

        hteamName = st.session_state.teams_dict[team_ids[0]]['home_team_name']
        ateamName = st.session_state.teams_dict[team_ids[1]]['away_team_name']

        # تصفية DataFrame لكل فريق
        home_df = df[df['teamId'].astype(int) == team_ids[0]]
        away_df = df[df['teamId'].astype(int) == team_ids[1]]

        # نسبة الاستحواذ
        hpossdf = home_df[home_df['type'] == 'Pass']
        apossdf = away_df[away_df['type'] == 'Pass']
        hposs = round((len(hpossdf) / (len(hpossdf) + len(apossdf))) * 100, 2) if (len(hpossdf) + len(apossdf)) > 0 else 0
        aposs = round((len(apossdf) / (len(hpossdf) + len(apossdf))) * 100, 2) if (len(hpossdf) + len(apossdf)) > 0 else 0

        # نسبة الميل الميداني (Field Tilt)
        hftdf = home_df[(home_df['isTouch'] == 1) & (home_df['x'] >= 70)]
        aftdf = away_df[(away_df['isTouch'] == 1) & (away_df['x'] >= 70)]
        hft = round((len(hftdf) / (len(hftdf) + len(aftdf))) * 100, 2) if (len(hftdf) + len(aftdf)) > 0 else 0
        aft = round((len(aftdf) / (len(hftdf) + len(aftdf))) * 100, 2) if (len(hftdf) + len(aftdf)) > 0 else 0

        # إجمالي التمريرات
        htotalPass = len(home_df[home_df['type'] == 'Pass'])
        atotalPass = len(away_df[away_df['type'] == 'Pass'])

        # التمريرات الناجحة
        hAccPass = len(home_df[(home_df['type'] == 'Pass') & (home_df['outcomeType'] == 'Successful')])
        aAccPass = len(away_df[(away_df['type'] == 'Pass') & (away_df['outcomeType'] == 'Successful')])

        # التمريرات الناجحة (بدون الثلث الدفاعي)
        hAccPasswdt = len(home_df[(home_df['type'] == 'Pass') & (home_df['outcomeType'] == 'Successful') & (home_df['endX'] > 35)])
        aAccPasswdt = len(away_df[(away_df['type'] == 'Pass') & (away_df['outcomeType'] == 'Successful') & (away_df['endX'] > 35)])

        # الكرات الطويلة
        hLongB = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_longball'] == True)])
        aLongB = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_longball'] == True)])

        # الكرات الطويلة الناجحة
        hAccLongB = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_longball'] == True) & (home_df['outcomeType'] == 'Successful')])
        aAccLongB = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_longball'] == True) & (away_df['outcomeType'] == 'Successful')])

        # العرضيات
        hCrss = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_cross'] == True)])
        aCrss = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_cross'] == True)])

        # العرضيات الناجحة
        hAccCrss = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_cross'] == True) & (home_df['outcomeType'] == 'Successful')])
        aAccCrss = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_cross'] == True) & (away_df['outcomeType'] == 'Successful')])

        # الركلات الحرة
        hfk = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_freekick'] == True)])
        afk = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_freekick'] == True)])

        # الركنيات
        hCor = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_corner'] == True)])
        aCor = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_corner'] == True)])

        # رميات التماس
        htins = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_throwin'] == True)])
        atins = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_throwin'] == True)])

        # ركلات المرمى
        hglkk = len(home_df[(home_df['type'] == 'Pass') & (home_df['has_goalkick'] == True)])
        aglkk = len(away_df[(away_df['type'] == 'Pass') & (away_df['has_goalkick'] == True)])

        # المراوغات
        htotalDrb = len(home_df[home_df['type'] == 'TakeOn'])
        atotalDrb = len(away_df[away_df['type'] == 'TakeOn'])

        # المراوغات الناجحة
        hAccDrb = len(home_df[(home_df['type'] == 'TakeOn') & (home_df['outcomeType'] == 'Successful')])
        aAccDrb = len(away_df[(away_df['type'] == 'TakeOn') & (away_df['outcomeType'] == 'Successful')])

        # متوسط طول ركلات المرمى
        home_goalkick = home_df[(home_df['type'] == 'Pass') & (home_df['has_goalkick'] == True)]
        away_goalkick = away_df[(away_df['type'] == 'Pass') & (away_df['has_goalkick'] == True)]

        if len(home_goalkick) != 0:
            try:
                home_goalkick['qualifiers'] = home_goalkick['qualifiers'].apply(ast.literal_eval)
                def extract_length(qualifiers):
                    for item in qualifiers:
                        if 'displayName' in item['type'] and item['type']['displayName'] == 'Length':
                            return float(item['value'])
                    return None
                home_goalkick['length'] = home_goalkick['qualifiers'].apply(extract_length).astype(float)
                hglkl = round(home_goalkick['length'].mean(), 2)
            except Exception as e:
                st.warning(f"خطأ في حساب طول ركلات المرمى للفريق المضيف: {e}")
                hglkl = 0
        else:
            hglkl = 0

        if len(away_goalkick) != 0:
            try:
                away_goalkick['qualifiers'] = away_goalkick['qualifiers'].apply(ast.literal_eval)
                def extract_length(qualifiers):
                    for item in qualifiers:
                        if 'displayName' in item['type'] and item['type']['displayName'] == 'Length':
                            return float(item['value'])
                    return None
                away_goalkick['length'] = away_goalkick['qualifiers'].apply(extract_length).astype(float)
                aglkl = round(away_goalkick['length'].mean(), 2)
            except Exception as e:
                st.warning(f"خطأ في حساب طول ركلات المرمى للفريق الضيف: {e}")
                aglkl = 0
        else:
            aglkl = 0

        # حساب PPDA
        home_def_acts = home_df[(home_df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle')) & (home_df['x'] > 35)]
        away_def_acts = away_df[(away_df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle')) & (away_df['x'] > 35)]
        home_pass = home_df[(home_df['type'] == 'Pass') & (home_df['outcomeType'] == 'Successful') & (home_df['x'] < 70)]
        away_pass = away_df[(away_df['type'] == 'Pass') & (away_df['outcomeType'] == 'Successful') & (away_df['x'] < 70)]
        home_ppda = round((len(away_pass) / len(home_def_acts)) if len(home_def_acts) > 0 else 0, 2)
        away_ppda = round((len(home_pass) / len(away_def_acts)) if len(away_def_acts) > 0 else 0, 2)

        # قائمة الإحصائيات
        stats = [
            (reshape_arabic_text('الاستحواذ %'), hposs, aposs, True),
            (reshape_arabic_text('الميل الميداني %'), hft, aft, True),
            (reshape_arabic_text('إجمالي التمريرات'), htotalPass, atotalPass, False),
            (reshape_arabic_text('التمريرات الناجحة'), hAccPass, aAccPass, False),
            (reshape_arabic_text('دقة التمرير %'), round((hAccPass/htotalPass)*100, 2) if htotalPass > 0 else 0, round((aAccPass/atotalPass)*100, 2) if atotalPass > 0 else 0, True),
            (reshape_arabic_text('التمريرات الناجحة (بدون الثلث الدفاعي)'), hAccPasswdt, aAccPasswdt, False),
            (reshape_arabic_text('الكرات الطويلة'), hLongB, aLongB, False),
            (reshape_arabic_text('الكرات الطويلة الناجحة'), hAccLongB, aAccLongB, False),
            (reshape_arabic_text('دقة الكرات الطويلة %'), round((hAccLongB/hLongB)*100, 2) if hLongB > 0 else 0, round((aAccLongB/aLongB)*100, 2) if aLongB > 0 else 0, True),
            (reshape_arabic_text('العرضيات'), hCrss, aCrss, False),
            (reshape_arabic_text('العرضيات الناجحة'), hAccCrss, aAccCrss, False),
            (reshape_arabic_text('دقة العرضيات %'), round((hAccCrss/hCrss)*100, 2) if hCrss > 0 else 0, round((aAccCrss/aCrss)*100, 2) if aCrss > 0 else 0, True),
            (reshape_arabic_text('الركلات الحرة'), hfk, afk, False),
            (reshape_arabic_text('الركنيات'), hCor, aCor, False),
            (reshape_arabic_text('رميات التماس'), htins, atins, False),
            (reshape_arabic_text('ركلات المرمى'), hglkk, aglkk, False),
            (reshape_arabic_text('متوسط طول ركلات المرمى'), hglkl, aglkl, False),
            (reshape_arabic_text('المراوغات'), htotalDrb, atotalDrb, False),
            (reshape_arabic_text('المراوغات الناجحة'), hAccDrb, aAccDrb, False),
            (reshape_arabic_text('نسبة نجاح المراوغات %'), round((hAccDrb/htotalDrb)*100, 2) if htotalDrb > 0 else 0, round((aAccDrb/atotalDrb)*100, 2) if atotalDrb > 0 else 0, True),
            (reshape_arabic_text('PPDA'), home_ppda, away_ppda, False)
        ]

        # إعداد الألوان
        bg_color = '#1e1e2f'
        text_color = '#ffffff'
        home_color = '#ff4d4d'
        away_color = '#4d79ff'
        gradient_colors = [home_color, '#ffffff', away_color]
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom_gradient", gradient_colors)

        # إعداد الرسم
        fig, ax = plt.subplots(figsize=(12, 14), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.axis('off')

        # إضافة خلفية متدرجة
        gradient = np.linspace(0, 1, 512)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.1, extent=[0, 12, 0, 14])

        # إضافة شعاري الناديين
        def add_logo(ax, image_url, x, y, zoom=0.15):
            try:
                response = requests.get(image_url)
                img = plt.imread(BytesIO(response.content))
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, boxcoords="axes fraction")
                ax.add_artist(ab)
            except Exception as e:
                st.warning(f"فشل تحميل الشعار: {str(e)}")

        home_team_logo = "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg"
        away_team_logo = "https://upload.wikimedia.org/wikipedia/en/2/2b/Inter_Milan_2014_logo.svg"
        add_logo(ax, home_team_logo, 0.25, 0.95)
        add_logo(ax, away_team_logo, 0.75, 0.95)

        # إضافة أسماء الفرق
        ax.text(0.25, 0.90, hteamName, color=home_color, fontsize=16, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.75, 0.90, ateamName, color=away_color, fontsize=16, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

        # إضافة العنوان
        ax.text(0.5, 0.85, reshape_arabic_text('إحصائيات المباراة'), color=text_color, fontsize=20, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

        # إعداد مواقع الإحصائيات
        y_positions = np.linspace(0.80, 0.05, len(stats))
        bar_width = 0.4

        for i, (stat_name, home_value, away_value, is_percentage) in enumerate(stats):
            y = y_positions[i]
            ax.text(0.5, y, stat_name, color=text_color, fontsize=12, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
            home_text = f"{home_value}{'%' if is_percentage else ''}"
            ax.text(0.25, y, home_text, color=home_color, fontsize=12, ha='center', va='center', transform=ax.transAxes)
            away_text = f"{away_value}{'%' if is_percentage else ''}"
            ax.text(0.75, y, away_text, color=away_color, fontsize=12, ha='center', va='center', transform=ax.transAxes)
            total = home_value + away_value
            if total > 0:
                home_ratio = home_value / total
                away_ratio = away_value / total
            else:
                home_ratio = 0.5
                away_ratio = 0.5
            bar_x_start = 0.35
            bar_x_end = 0.65
            bar_length = bar_x_end - bar_x_start
            bar_gradient = np.linspace(0, 1, 256)
            bar_gradient = np.vstack((bar_gradient, bar_gradient))
            ax.imshow(bar_gradient, aspect='auto', cmap=cmap, extent=[bar_x_start, bar_x_end, y-0.01, y+0.01], transform=ax.transAxes, alpha=0.8)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"خطأ في عرض إحصائيات المباراة: {str(e)}")
        st.write("أعمدة DataFrame:", df.columns.tolist())
        st.write("Team IDs:", team_ids)
        st.exception(e)
# تبويب تصور التمريرات
with tab4:
    st.subheader(reshape_arabic_text("تصور التمريرات"))
    try:
        df = st.session_state.df
        team_ids = list(st.session_state.teams_dict.keys())
        hteamName = st.session_state.teams_dict[team_ids[0]]['home_team_name']
        ateamName = st.session_state.teams_dict[team_ids[1]]['away_team_name']
        hcol = st.session_state.teams_dict[team_ids[0]].get('home_team_kit_colour', '#ff0000')
        acol = st.session_state.teams_dict[team_ids[1]].get('away_team_kit_colour', '#0000ff')
        team_options = ['All', hteamName, ateamName]
        selected_team = st.selectbox(reshape_arabic_text("اختر الفريق"), team_options, index=0)
        filtered_df = df[df['type'] == 'Pass'].copy()
        if selected_team != 'All':
            team_id = team_ids[0] if selected_team == hteamName else team_ids[1]
            filtered_df = filtered_df[filtered_df['teamId'] == team_id]
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        try:
            pitch_img = plt.imread('pitch.png')
        except FileNotFoundError:
            st.error("صورة 'pitch.png' غير موجودة. يرجى إضافتها إلى نفس المجلد.")
            st.stop()
        ax.imshow(pitch_img, extent=[0, 105, 0, 68])
        for _, row in filtered_df.iterrows():
            if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['endX']) and pd.notna(row['endY']):
                color = hcol if row['teamId'] == team_ids[0] else acol
                ax.plot([row['x'], row['endX']], [row['y'], row['endY']], color=color, alpha=0.5)
                ax.scatter(row['x'], row['y'], color=color, s=50, alpha=0.7)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
    except Exception as e:
        st.error(f"خطأ في تصور التمريرات: {str(e)}")
        st.write("أعمدة DataFrame:", df.columns.tolist())
        st.exception(e)
