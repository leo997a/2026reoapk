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

    # استخراج أحداث الفريق في الثلث الهجومي فقط (x >= 66.7) - صحيح لنظام 0-100
    team_events = df[df['teamId'] == team_id]
    final_third_events = team_events[team_events['x'] >= 66.7]

    # تقسيم الثلث الهجومي إلى ثلاثة مناطق عرضية (باستخدام نظام 0-100)
    left_zone = final_third_events[final_third_events['y'] <= 33.3]
    center_zone = final_third_events[(final_third_events['y'] > 33.3) & (final_third_events['y'] <= 66.7)]
    right_zone = final_third_events[final_third_events['y'] > 66.7]

    # حساب النسب المئوية
    total = len(left_zone) + len(center_zone) + len(right_zone)
    left_pct = (len(left_zone) / total * 100) if total > 0 else 0
    center_pct = (len(center_zone) / total * 100) if total > 0 else 0
    right_pct = (len(right_zone) / total * 100) if total > 0 else 0

    # تحديد المنطقة ذات النسبة الأعلى لجعلها أكثر وضوحًا (أقل شفافية)
    percentages = [left_pct, center_pct, right_pct]
    if total > 0:
        max_pct_index = percentages.index(max(percentages))
        min_pct_index = percentages.index(min(percentages))
        # Handle cases where max and min are the same (e.g., all zones 0% or all 33.3%)
        if max_pct_index == min_pct_index:
             mid_pct_index = (max_pct_index + 1) % 3 # Assign arbitrarily if all equal
        else:
             mid_pct_index = 3 - max_pct_index - min_pct_index
    else:
        max_pct_index, mid_pct_index, min_pct_index = 0, 1, 2 # Default if no events

    # تعيين قيم الشفافية بناءً على النسب (المنطقة ذات النسبة الأعلى تكون أقل شفافية)
    alpha_values = [0.7, 0.7, 0.7]  # قيم افتراضية
    alpha_values[max_pct_index] = 0.9  # أقل شفافية للمنطقة ذات النسبة الأعلى
    alpha_values[mid_pct_index] = 0.7  # شفافية متوسطة
    alpha_values[min_pct_index] = 0.5  # أكثر شفافية للمنطقة ذات النسبة الأقل

    # رسم الملعب (باستخدام pitch_type='opta')
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    # استخدمنا opta ليتوافق مع بيانات WhoScored (0-100)
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color, stripe=False, goal_type='box')
    pitch.draw(ax=ax)

    # تحديد اتجاه الهجوم (من اليسار إلى اليمين)
    attack_direction = 'right'  # يمكن تغييره إلى 'left' حسب الحاجة

    # تلوين كامل الثلث الهجومي (باستخدام نظام 0-100)
    if attack_direction == 'right':
        # تلوين الثلث الهجومي بالكامل أولاً (x=66.7 to 100, y=0 to 100)
        ax.add_patch(patches.Rectangle((66.7, 0), 33.3, 100,
                                     facecolor=team_color, alpha=0.3,
                                     edgecolor='white', linewidth=0.5, zorder=1))

        # ثم تلوين المناطق الثلاثة بشفافية مختلفة
        # يسار (y=0 to 33.3)
        ax.add_patch(patches.Rectangle((66.7, 0), 33.3, 33.3,
                                     facecolor=team_color, alpha=alpha_values[0],
                                     edgecolor='white', linewidth=0.5, zorder=2))
        # وسط (y=33.3 to 66.7)
        ax.add_patch(patches.Rectangle((66.7, 33.3), 33.3, 33.4,
                                     facecolor=team_color, alpha=alpha_values[1],
                                     edgecolor='white', linewidth=0.5, zorder=2))
        # يمين (y=66.7 to 100)
        ax.add_patch(patches.Rectangle((66.7, 66.7), 33.3, 33.3,
                                     facecolor=team_color, alpha=alpha_values[2],
                                     edgecolor='white', linewidth=0.5, zorder=2))
    else: # Attack direction left
        # تلوين الثلث الهجومي بالكامل أولاً (x=0 to 33.3, y=0 to 100)
        ax.add_patch(patches.Rectangle((0, 0), 33.3, 100,
                                     facecolor=team_color, alpha=0.3,
                                     edgecolor='white', linewidth=0.5, zorder=1))

        # ثم تلوين المناطق الثلاثة بشفافية مختلفة
        # يسار (y=0 to 33.3)
        ax.add_patch(patches.Rectangle((0, 0), 33.3, 33.3,
                                     facecolor=team_color, alpha=alpha_values[0],
                                     edgecolor='white', linewidth=0.5, zorder=2))
        # وسط (y=33.3 to 66.7)
        ax.add_patch(patches.Rectangle((0, 33.3), 33.3, 33.4,
                                     facecolor=team_color, alpha=alpha_values[1],
                                     edgecolor='white', linewidth=0.5, zorder=2))
        # يمين (y=66.7 to 100)
        ax.add_patch(patches.Rectangle((0, 66.7), 33.3, 33.3,
                                     facecolor=team_color, alpha=alpha_values[2],
                                     edgecolor='white', linewidth=0.5, zorder=2))

    # إضافة دوائر خلف النسب لتحسين الوضوح (تعديل الإحداثيات لـ 0-100)
    if attack_direction == 'right':
        center_x = 83.3 # مركز الثلث الهجومي أفقيًا
        # يسار (مركز y = 16.65)
        circle1 = plt.Circle((center_x, 16.65), 10, color='white', alpha=0.2, zorder=3)
        ax.add_artist(circle1)
        ax.text(center_x, 16.65, f"{left_pct:.1f}%", color='white', fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        # وسط (مركز y = 50)
        circle2 = plt.Circle((center_x, 50), 10, color='white', alpha=0.2, zorder=3)
        ax.add_artist(circle2)
        ax.text(center_x, 50, f"{center_pct:.1f}%", color='white', fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        # يمين (مركز y = 83.35)
        circle3 = plt.Circle((center_x, 83.35), 10, color='white', alpha=0.2, zorder=3)
        ax.add_artist(circle3)
        ax.text(center_x, 83.35, f"{right_pct:.1f}%", color='white', fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    else: # Attack direction left
        center_x = 16.7 # مركز الثلث الهجومي أفقيًا
        # يسار (مركز y = 16.65)
        circle1 = plt.Circle((center_x, 16.65), 10, color='white', alpha=0.2, zorder=3)
        ax.add_artist(circle1)
        ax.text(center_x, 16.65, f"{left_pct:.1f}%", color='white', fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        # وسط (مركز y = 50)
        circle2 = plt.Circle((center_x, 50), 10, color='white', alpha=0.2, zorder=3)
        ax.add_artist(circle2)
        ax.text(center_x, 50, f"{center_pct:.1f}%", color='white', fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        # يمين (مركز y = 83.35)
        circle3 = plt.Circle((center_x, 83.35), 10, color='white', alpha=0.2, zorder=3)
        ax.add_artist(circle3)
        ax.text(center_x, 83.35, f"{right_pct:.1f}%", color='white', fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    # --- باقي الكود (الأسهم، الشعارات، العناوين، العلامة المائية) يبقى كما هو ---
    # --- تأكد من أن إحداثيات هذه العناصر مناسبة للملعب 0-100 ---

    # مثال لتعديل سهم اتجاه الهجوم
    if attack_direction == 'right':
        ax.arrow(40, 105, 20, 0, head_width=4, head_length=4, fc='white', ec='white', zorder=5, linewidth=2, alpha=0.8)
        ax.text(50, 110, reshape_arabic_text("اتجاه الهجوم"), color='white', fontsize=10,
                fontweight='bold', ha='center', va='center', zorder=5,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    else:
        ax.arrow(60, 105, -20, 0, head_width=4, head_length=4, fc='white', ec='white', zorder=5, linewidth=2, alpha=0.8)
        ax.text(50, 110, reshape_arabic_text("اتجاه الهجوم"), color='white', fontsize=10,
                fontweight='bold', ha='center', va='center', zorder=5,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])

    # تعديل موضع الشعارات والنصوص لتناسب الملعب الجديد
    # محاولة تحميل شعارات الفرق من الملفات المحلية أو من الإنترنت
    try:
        home_logo_path = f"logos/{json_data['home']['teamId']}.png"
        away_logo_path = f"logos/{json_data['away']['teamId']}.png"

        if os.path.exists(home_logo_path):
            home_logo = plt.imread(home_logo_path)
            # تعديل الموضع النسبي للشعار
            home_logo_ax = fig.add_axes([0.1, 0.88, 0.08, 0.08], anchor='NW', zorder=10)
            home_logo_ax.imshow(home_logo)
            home_logo_ax.axis('off')
        else:
            home_logo_circle = plt.Circle((15, 90), 5, color=hcol, ec='white', linewidth=1, zorder=10)
            ax.add_patch(home_logo_circle)

        if os.path.exists(away_logo_path):
            away_logo = plt.imread(away_logo_path)
            # تعديل الموضع النسبي للشعار
            away_logo_ax = fig.add_axes([0.82, 0.88, 0.08, 0.08], anchor='NE', zorder=10)
            away_logo_ax.imshow(away_logo)
            away_logo_ax.axis('off')
        else:
            away_logo_circle = plt.Circle((85, 90), 5, color=acol, ec='white', linewidth=1, zorder=10)
            ax.add_patch(away_logo_circle)

    except Exception as e:
        st.warning(f"لم يتم تحميل شعارات الفرق: {e}")
        home_logo_circle = plt.Circle((15, 90), 5, color=hcol, ec='white', linewidth=1, zorder=10)
        away_logo_circle = plt.Circle((85, 90), 5, color=acol, ec='white', linewidth=1, zorder=10)
        ax.add_patch(home_logo_circle)
        ax.add_patch(away_logo_circle)

    # إضافة أسماء الفرق بجانب الشعارات (تعديل الموضع)
    ax.text(25, 90, json_data['home']['name'], fontsize=14, color='white', ha='left', va='center',
             fontweight='bold', zorder=10, path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
    ax.text(75, 90, json_data['away']['name'], fontsize=14, color='white', ha='right', va='center',
             fontweight='bold', zorder=10, path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])

    # إضافة خلفية للعنوان (تعديل الموضع النسبي)
    title_bg = patches.Rectangle((0.25, 0.95), 0.5, 0.05, transform=fig.transFigure,
                               facecolor='#003366', alpha=0.8, edgecolor='white',
                               linewidth=1, zorder=9)
    fig.patches.append(title_bg)

    # إضافة العنوان الرئيسي
    title_text = f"{json_data['home']['name']} - {json_data['away']['name']}"
    fig.text(0.5, 0.975, title_text, fontsize=18, color='white', ha='center',
             fontweight='bold', zorder=10,
             path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    # إضافة العنوان الفرعي
    subtitle_text = reshape_arabic_text('تحليل مناطق الهجوم')
    fig.text(0.5, 0.955, subtitle_text, fontsize=14, color='white', ha='center', zorder=10,
             path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])

    # إضافة العلامة المائية إذا كانت مفعلة
    if watermark_enabled:
        add_watermark(fig, text=watermark_text, alpha=watermark_opacity,
                     fontsize=watermark_size, color=watermark_color,
                     x_pos=watermark_x, y_pos=watermark_y,
                     ha=watermark_ha, va=watermark_va)

    # ضبط التخطيط
    plt.tight_layout(rect=[0, 0.05, 1, 0.93]) # تعديل rect لتوفير مساحة للعناصر العلوية والسفلية
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
# دالة لرسم إحصائيات المباراة
def plot_match_stats(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color, watermark_enabled, watermark_text, watermark_opacity, watermark_size, watermark_color, watermark_x, watermark_y, watermark_ha, watermark_va):
    """
    رسم إحصائيات المباراة للفريقين مع تصميم عربي
    
    Parameters:
    ax: matplotlib.axes.Axes
        المحور لرسم الإحصائيات
    df: pd.DataFrame
        إطار بيانات الأحداث
    hteamName, ateamName: str
        أسماء الفريقين (المضيف والضيف)
    hcol, acol: str
        ألوان الفريقين
    bg_color, line_color: str
        لون الخلفية ولون الخطوط
    watermark_enabled: bool
        تفعيل العلامة المائية
    watermark_text: str
        نص العلامة المائية
    watermark_opacity: float
        شفافية العلامة المائية
    watermark_size: int
        حجم العلامة المائية
    watermark_color: str
        لون العلامة المائية
    watermark_x, watermark_y: float
        موقع العلامة المائية
    watermark_ha, watermark_va: str
        محاذاة العلامة المائية
    
    Returns:
    pd.DataFrame
        إطار بيانات يحتوي على إحصائيات الفريقين
    """
    # إعداد الملعب
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-5, 68.5)

    # حساب الإحصائيات
    # الاستحواذ
    hpossdf = df[(df['teamName'] == hteamName) & (df['type'] == 'Pass')]
    apossdf = df[(df['teamName'] == ateamName) & (df['type'] == 'Pass')]
    total_poss = len(hpossdf) + len(apossdf)
    hposs = round((len(hpossdf) / total_poss * 100) if total_poss > 0 else 0, 2)
    aposs = round((len(apossdf) / total_poss * 100) if total_poss > 0 else 0, 2)

    # ميلان الملعب (Field Tilt)
    hftdf = df[(df['teamName'] == hteamName) & (df['isTouch'] == True) & (df['x'] >= 70)]
    aftdf = df[(df['teamName'] == ateamName) & (df['isTouch'] == True) & (df['x'] >= 70)]
    total_ft = len(hftdf) + len(aftdf)
    hft = round((len(hftdf) / total_ft * 100) if total_ft > 0 else 0, 2)
    aft = round((len(aftdf) / total_ft * 100) if total_ft > 0 else 0, 2)

    # إجمالي التمريرات
    htotalPass = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Pass')])
    atotalPass = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Pass')])

    # التمريرات الناجحة
    hAccPass = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')])
    aAccPass = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')])

    # الكرات الطويلة
    hLongB = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Longball', na=False)) & (~df['qualifiers'].str.contains('Corner|Cross', na=False))])
    aLongB = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Longball', na=False)) & (~df['qualifiers'].str.contains('Corner|Cross', na=False))])

    # الكرات الطويلة الناجحة
    hAccLongB = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Longball', na=False)) & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Corner|Cross', na=False))])
    aAccLongB = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Longball', na=False)) & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Corner|Cross', na=False))])

    # التدخلات
    htkl = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Tackle')])
    atkl = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Tackle')])

    # التدخلات الناجحة
    htklw = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Tackle') & (df['outcomeType'] == 'Successful')])
    atklw = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Tackle') & (df['outcomeType'] == 'Successful')])

    # الاعتراضات
    hintc = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Interception')])
    aintc = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Interception')])

    # التشتيتات
    hclr = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Clearance')])
    aclr = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Clearance')])

    # المواجهات الهوائية
    harl = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Aerial')])
    aarl = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Aerial')])

    # المواجهات الهوائية الناجحة
    harlw = len(df[(df['teamName'] == hteamName) & (df['type'] == 'Aerial') & (df['outcomeType'] == 'Successful')])
    aarlw = len(df[(df['teamName'] == ateamName) & (df['type'] == 'Aerial') & (df['outcomeType'] == 'Successful')])

    # PPDA (معادلة مبسطة)
    home_def_acts = df[(df['teamName'] == hteamName) & (df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle', na=False)) & (df['x'] > 35)]
    away_def_acts = df[(df['teamName'] == ateamName) & (df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle', na=False)) & (df['x'] > 35)]
    home_pass = df[(df['teamName'] == hteamName) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['x'] < 70)]
    away_pass = df[(df['teamName'] == ateamName) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['x'] < 70)]
    home_ppda = round(len(away_pass) / len(home_def_acts), 2) if len(home_def_acts) > 0 else 0
    away_ppda = round(len(home_pass) / len(away_def_acts), 2) if len(away_def_acts) > 0 else 0

    # متوسط التمريرات لكل تسلسل
    pass_df_home = df[(df['type'] == 'Pass') & (df['teamName'] == hteamName)]
    pass_counts_home = pass_df_home.groupby('possession_id').size()
    PPS_home = round(pass_counts_home.mean()) if not pass_counts_home.empty else 0
    pass_df_away = df[(df['type'] == 'Pass') & (df['teamName'] == ateamName)]
    pass_counts_away = pass_df_away.groupby('possession_id').size()
    PPS_away = round(pass_counts_away.mean()) if not pass_counts_away.empty else 0

    # عدد التسلسلات التي تحتوي على 10+ تمريرات
    pass_seq_10_more_home = pass_counts_home[pass_counts_home >= 10].count() if not pass_counts_home.empty else 0
    pass_seq_10_more_away = pass_counts_away[pass_counts_away >= 10].count() if not pass_counts_away.empty else 0

    # إعداد الرسم
    # صندوق العنوان
    head_y = [62, 68, 68, 62]
    head_x = [0, 0, 105, 105]
    ax.fill(head_x, head_y, '#003366', alpha=0.8)  # لون أزرق داكن بدلاً من البرتقالي
    ax.text(52.5, 64.5, reshape_arabic_text("إحصائيات المباراة"), ha='center', va='center', color='white', fontsize=25, fontweight='bold', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    # إعداد الأشرطة
    stats_title = [58, 58-(1*6), 58-(2*6), 58-(3*6), 58-(4*6), 58-(5*6), 58-(6*6), 58-(7*6), 58-(8*6), 58-(9*6), 58-(10*6)]  # مواقع الأشرطة على المحور y
    stats_home = [hposs, hft, htotalPass, hLongB, htkl, hintc, hclr, harl, home_ppda, PPS_home, pass_seq_10_more_home]
    stats_away = [aposs, aft, atotalPass, aLongB, atkl, aintc, aclr, aarl, away_ppda, PPS_away, pass_seq_10_more_away]

    # تطبيع القيم للأشرطة
    stats_normalized_home = []
    stats_normalized_away = []
    for h, a in zip(stats_home, stats_away):
        total = h + a
        if total > 0:
            stats_normalized_home.append(-(h / total) * 50)
            stats_normalized_away.append((a / total) * 50)
        else:
            stats_normalized_home.append(0)
            stats_normalized_away.append(0)

    start_x = 52.5
    ax.barh(stats_title, stats_normalized_home, height=4, color=hcol, left=start_x, alpha=0.9)
    ax.barh(stats_title, stats_normalized_away, height=4, color=acol, left=start_x, alpha=0.9)

    # إيقاف عناصر المحاور
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # النصوص (العناوين)
    stat_labels = [
        "الاستحواذ",
        "ميلان الملعب",
        "التمريرات (الناجحة)",
        "الكرات الطويلة (الناجحة)",
        "التدخلات (الناجحة)",
        "الاعتراضات",
        "التشتيتات",
        "المواجهات الهوائية (الناجحة)",
        "PPDA",
        "متوسط التمريرات/التسلسل",
        "تسلسلات 10+ تمريرات"
    ]
    for i, label in enumerate(stat_labels):
        ax.text(52.5, stats_title[i], reshape_arabic_text(label), color='white', fontsize=14, ha='center', va='center', fontweight='bold', path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])

    # النصوص (القيم)
    ax.text(0, 58, f"{round(hposs)}%", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(1*6), f"{round(hft)}%", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(2*6), f"{htotalPass} ({hAccPass})", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(3*6), f"{hLongB} ({hAccLongB})", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(4*6), f"{htkl} ({htklw})", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(5*6), f"{hintc}", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(6*6), f"{hclr}", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(7*6), f"{harl} ({harlw})", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(8*6), f"{home_ppda}", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(9*6), f"{int(PPS_home)}", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(0, 58-(10*6), f"{pass_seq_10_more_home}", color=line_color, fontsize=16, ha='right', va='center', fontweight='bold')

    ax.text(105, 58, f"{round(aposs)}%", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(1*6), f"{round(aft)}%", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(2*6), f"{atotalPass} ({aAccPass})", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(3*6), f"{aLongB} ({aAccLongB})", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(4*6), f"{atkl} ({atklw})", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(5*6), f"{aintc}", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(6*6), f"{aclr}", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(7*6), f"{aarl} ({aarlw})", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(8*6), f"{away_ppda}", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(9*6), f"{int(PPS_away)}", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(105, 58-(10*6), f"{pass_seq_10_more_away}", color=line_color, fontsize=16, ha='left', va='center', fontweight='bold')

    # إنشاء إطار بيانات الإحصائيات
    home_data = {
        'اسم الفريق': hteamName,
        'الاستحواذ (%)': hposs,
        'ميلان الملعب (%)': hft,
        'إجمالي التمريرات': htotalPass,
        'التمريرات الناجحة': hAccPass,
        'الكرات الطويلة': hLongB,
        'الكرات الطويلة الناجحة': hAccLongB,
        'التدخلات': htkl,
        'التدخلات الناجحة': htklw,
        'الاعتراضات': hintc,
        'التشتيتات': hclr,
        'المواجهات الهوائية': harl,
        'المواجهات هائية الناجحة': harlw,
        'PPDA': home_ppda,
        'متوسط التمريرات/التسلسل': PPS_home,
        'تسلسلات 10+ تمريرات': pass_seq_10_more_home
    }

    away_data = {
        'اسم الفريق': ateamName,
        'الاستحواذ (%)': aposs,
        'ميلان الملعب (%)': aft,
        'إجمالي التمريرات': atotalPass,
        'التمريرات الناجحة': aAccPass,
        'الكرات الطويلة': aLongB,
        'الكرات الطويلة الناجحة': aAccLongB,
        'التدخلات': atkl,
        'التدخلات الناجحة': atklw,
        'الاعتراضات': aintc,
        'التشتيتات': aclr,
        'المواجهات الهوائية': aarl,
        'المواجهات هائية الناجحة': aarlw,
        'PPDA': away_ppda,
        'متوسط التمريرات/التسلسل': PPS_away,
        'تسلسلات 10+ تمريرات': pass_seq_10_more_away
    }

    return pd.DataFrame([home_data, away_data])

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

    # علامات التبويب
    try:
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
    selected_period = period_map[period_choice]

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
    selected_region = region_map[region_choice]

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
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"خطأ في حساب PPDA: {str(e)}")
        st.write("يرجى التحقق من البيانات المحملة.")

with tab2:
    st.subheader(reshape_arabic_text("التبويب الثاني"))
    # إنشاء علامتي تبويب وهميتين داخل Tab 2
    dummy_tab1, dummy_tab2 = st.tabs([reshape_arabic_text("بيانات وهمية 1"), 
                                      reshape_arabic_text("بيانات وهمية 2")])
    
    # محتوى التبويب الوهمي الأول
    with dummy_tab1:
        st.write(reshape_arabic_text("هذا تبويب وهمي 1 داخل التبويب الثاني."))
        st.markdown(reshape_arabic_text("يمكنك إضافة أي محتوى هنا، مثل نصوص أو رسوم بيانية."))
        # مثال لبيانات وهمية
        dummy_data1 = pd.DataFrame({
            reshape_arabic_text("الفريق"): [hteamName, ateamName],
            reshape_arabic_text("الأهداف"): [3, 2]
        })
        st.dataframe(dummy_data1, use_container_width=True)
    
    # محتوى التبويب الوهمي الثاني
    with dummy_tab2:
        st.write(reshape_arabic_text("هذا تبويب وهمي 2 داخل التبويب الثاني."))
        st.markdown(reshape_arabic_text("مثال آخر لمحتوى وهمي."))
        # مثال لرسم بياني وهمي
        fig, ax = plt.subplots()
        ax.bar([reshape_arabic_text(hteamName), reshape_arabic_text(ateamName)], [5, 3], color=[hcol, acol])
        ax.set_title(reshape_arabic_text("مقارنة وهمية"))
        st.pyplot(fig)

with tab3:
    st.subheader(reshape_arabic_text("إحصائيات المباراة"))
    
    # إنشاء الرسم
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=bg_color, dpi=150)
    
    try:
        # استدعاء دالة plot_match_stats
        stats_df = plot_match_stats(
            ax,
            st.session_state.df,
            hteamName,
            ateamName,
            hcol,
            acol,
            bg_color,
            line_color,
            watermark_enabled,
            watermark_text,
            watermark_opacity,
            watermark_size,
            watermark_color,
            watermark_x,
            watermark_y,
            watermark_ha,
            watermark_va
        )
        
        # إضافة العلامة المائية إذا كانت مفعلة
        if watermark_enabled:
            fig = add_watermark(fig, text=watermark_text, alpha=watermark_opacity, 
                               fontsize=watermark_size, color=watermark_color,
                               x_pos=watermark_x, y_pos=watermark_y, 
                               ha=watermark_ha, va=watermark_va)
        
        st.pyplot(fig)
        
        # عرض إطار البيانات
        st.subheader(reshape_arabic_text("تفاصيل الإحصائيات"))
        st.dataframe(stats_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"خطأ في عرض إحصائيات المباراة: {str(e)}")
