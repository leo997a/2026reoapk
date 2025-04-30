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


def attack_zones_analysis(fig, ax, hteamName, ateamName, hcol, acol, hteamID, ateamID):
    # تصفية الأحداث للفريق المحدد
    team_events = data[data['team_name'] == team_name]
    
    # تصفية التمريرات التي تنتهي في الثلث الأخير (x >= 80)
    passes = team_events[team_events['type_name'] == 'Pass']
    final_third_passes = passes[passes['pass_end_location'].apply(lambda loc: loc[0] >= 80 if isinstance(loc, list) else False)]
    
    # تقسيم الثلث الأخير إلى 3 مناطق عرضية (يسار، وسط، يمين)
    left_zone = final_third_passes[final_third_passes['pass_end_location'].apply(lambda loc: loc[1] < 26.67)]
    center_zone = final_third_passes[final_third_passes['pass_end_location'].apply(lambda loc: 26.67 <= loc[1] < 53.33)]
    right_zone = final_third_passes[final_third_passes['pass_end_location'].apply(lambda loc: loc[1] >= 53.33)]
    
    # حساب عدد التمريرات في كل منطقة
    zones = {
        'يسار': len(left_zone),
        'وسط': len(center_zone),
        'يمين': len(right_zone)
    }
    
    # تحديد المنطقة الأكثر هجومًا
    most_attacked_zone = max(zones, key=zones.get)
    max_attacks = zones[most_attacked_zone]
    
    # ألوان مختلفة لكل منطقة
    zone_colors = {
        'يسار': 'red',
        'وسط': 'green',
        'يمين': 'blue'
    }
    
    # مركز كل منطقة (لتحديد نقطة بداية السهم)
    zone_centers = {
        'يسار': (100, 13.33),
        'وسط': (100, 40),
        'يمين': (100, 66.67)
    }
    
    # رسم الأسهم لكل منطقة
    for zone, count in zones.items():
        if count > 0:  # رسم السهم فقط إذا كان هناك تمريرات
            center_x, center_y = zone_centers[zone]
            arrow_length = (count / max_attacks) * 15  # تكبير السهم بنسبة عدد التمريرات
            arrow_props = dict(facecolor=zone_colors[zone], edgecolor='black', width=2, headwidth=6, headlength=6)
            arrow = ax.add_patch(Arrow(center_x, center_y, arrow_length, 0, **arrow_props))
            # إضافة ظل للسهم
            arrow.set_path_effects([withStroke(linewidth=4, foreground='black', alpha=0.3)])
            
            # إضافة نص يوضح عدد التمريرات
            text = ax.text(center_x, center_y + 5, f"{count} تمريرة", color='white', fontsize=12, ha='center',
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            text.set_path_effects([withStroke(linewidth=2, foreground='white', alpha=0.5)])
    
    # عنوان الرسم
    title = reshape_arabic_text(f"مناطق الهجوم لفريق {team_name}")
    title_text = ax.text(60, 130, title, color='white', fontsize=16, ha='center', weight='bold')
    title_text.set_path_effects([withStroke(linewidth=3, foreground='black', alpha=0.7)])
    
    return zones, most_attacked_zone

import pandas as pd
import numpy as np
import streamlit as st

import pandas as pd
import numpy as np
import streamlit as st

def fetch_sofascore_ppda(match_url: str, team: str) -> float:
    """
    استخراج PPDA من Sofascore باستخدام رابط المباراة واسم الفريق.
    
    Args:
        match_url (str): رابط مباراة Sofascore (مثل https://www.sofascore.com/celta-vigo-barcelona/xxx)
        team (str): اسم الفريق (مثل "Celta Vigo")
    
    Returns:
        float: قيمة PPDA للفريق، أو None إذا فشل الاستخراج
    """
    try:
        # استخراج معرف المباراة من الرابط
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(match_url, headers=headers)
        if response.status_code != 200:
            st.warning(f"فشل الوصول إلى رابط Sofascore: {match_url}")
            return None
        
        # استخراج معرف المباراة
        soup = BeautifulSoup(response.content, "html.parser")
        id_tag = soup.select_one('link[href*="android-app:"]')
        if not id_tag:
            st.warning("لم يتم العثور على معرف المباراة في رابط Sofascore.")
            return None
        match_id = id_tag["href"].split("/")[-1]
        
        # الوصول إلى إحصائيات المباراة
        stats_url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
        stats_response = requests.get(stats_url, headers=headers)
        if stats_response.status_code != 200:
            st.warning(f"فشل الوصول إلى إحصائيات Sofascore للمباراة {match_id}")
            return None
        
        stats_data = json.loads(stats_response.content)
        if not stats_data or 'statistics' not in stats_data:
            st.warning("لا توجد بيانات إحصائيات متاحة من Sofascore.")
            return None
        
        # البحث عن PPDA في الإحصائيات
        ppda_value = None
        for stat_group in stats_data['statistics']:
            for item in stat_group['groups']:
                if item.get('groupName') == 'Team stats':
                    for stat in item['statisticsItems']:
                        if stat.get('name') == 'PPDA':
                            # PPDA قد يكون للفريق المنزلي أو الزائر
                            home_team = stats_data['homeTeam']['name']
                            away_team = stats_data['awayTeam']['name']
                            if team.lower() in home_team.lower():
                                ppda_value = float(stat['homeValue'])
                            elif team.lower() in away_team.lower():
                                ppda_value = float(stat['awayValue'])
                            break
                    if ppda_value is not None:
                        break
            if ppda_value is not None:
                break
        
        if ppda_value is None:
            st.warning(f"لم يتم العثور على PPDA لفريق {team} في بيانات Sofascore.")
            return None
        
        st.write(f"تم استخراج PPDA من Sofascore لفريق {team}: {ppda_value:.2f}")
        return ppda_value
    
    except Exception as e:
        st.error(f"خطأ أثناء استخراج PPDA من Sofascore: {str(e)}")
        return None

def calculate_team_ppda(
    events_df: pd.DataFrame,
    team: str,
    region: str = 'opponent_defensive_third',
    pitch_units: float = 105,
    period: str = None,
    min_def_actions: int = 2,
    min_pass_distance: float = 1.5,
    sofascore_match_url: str = None  # إضافة معلمة لرابط Sofascore
) -> dict:
    try:
        # نسخ إطار البيانات والتحقق من الإحداثيات
        df = events_df.copy()
        for col in ['x', 'y', 'endX', 'endY']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, pitch_units if col in ['x', 'endX'] else 68)
                if df[col].isna().any():
                    df = df.dropna(subset=[col])

        # التحقق من جودة البيانات
        if df.empty or 'type' not in df.columns or 'teamName' not in df.columns:
            st.error(f"البيانات غير صالحة لفريق {team}: إطار بيانات فارغ أو ينقصه أعمدة أساسية.")
            return {}

        # تصفية الفترة
        if period:
            df = df[df['period'] == period]
            if df.empty:
                st.warning(f"لا توجد بيانات للفترة {period} لفريق {team}.")
                return {}

        # الكشف عن تبديل الجوانب
        swap_sides = False
        if not period:
            first_half_x = df[df['period'] == 'FirstHalf']['x'].mean()
            second_half_x = df[df['period'] == 'SecondHalf']['x'].mean()
            if pd.notna(first_half_x) and pd.notna(second_half_x) and abs(first_half_x - second_half_x) > pitch_units / 3:
                swap_sides = True
                df.loc[df['period'] == 'SecondHalf', 'x'] = pitch_units - df.loc[df['period'] == 'SecondHalf', 'x']
                df.loc[df['period'] == 'SecondHalf', 'y'] = 68 - df.loc[df['period'] == 'SecondHalf', 'y']
                if 'endX' in df.columns and 'endY' in df.columns:
                    df.loc[df['period'] == 'SecondHalf', 'endX'] = pitch_units - df.loc[df['period'] == 'SecondHalf', 'endX']
                    df.loc[df['period'] == 'SecondHalf', 'endY'] = 68 - df.loc[df['period'] == 'SecondHalf', 'endY']
                st.write(f"تم تبديل الجوانب تلقائيًا للشوط الثاني لفريق {team}.")

        # تحديد الأفعال الدفاعية
        defs = ['Tackle', 'Interception', 'BlockedPass', 'Challenge', 'BallRecovery']
        if 'Pressure' in df['type'].unique():
            defs.append('Pressure')
        extended_defs = ['ShieldBallOpp']
        defs.extend([d for d in extended_defs if d in df['type'].unique()])
        st.write(f"أنواع الأفعال الدفاعية المستخدمة لـ {team}: {defs}")

        # تحديد المنطقة
        team_id = [k for k, v in st.session_state.get('teams_dict', {}).items() if v == team]
        team_id = team_id[0] if team_id else team
        is_home_team = team_id == min(st.session_state.get('teams_dict', {}).keys(), default=team)
        if region == 'opponent_defensive_third':
            x_min = pitch_units * (2 / 3) if is_home_team else 0
            x_max = pitch_units if is_home_team else pitch_units / 3
        else:
            raise ValueError(f"المنطقة غير مدعومة تلقائيًا: {region}")

        # تحليل انتشار الفريق واكتشاف الضغط العالي
        team_events = df[(df['teamName'] == team) & (df['x'] >= x_min) & (df['x'] <= x_max)]
        high_press = False
        if not team_events.empty:
            region_events = len(team_events)
            total_events = len(df[df['teamName'] == team])
            high_press_ratio = region_events / total_events if total_events > 0 else 0
            high_press = high_press_ratio > 0.15
            st.write(f"نسبة الأحداث في الثلث الدفاعي لـ {team}: {high_press_ratio:.2f}. الضغط العالي: {high_press}")
        else:
            high_press_ratio = 0
            st.warning(f"لا توجد أحداث كافية في الثلث الدفاعي لـ {team}.")
            return {}

        # حساب فقدان الكرة
        opponent = [t for t in df['teamName'].unique() if t != team][0] if len([t for t in df['type'].unique() if t != team]) > 0 else None
        if opponent:
            ball_losses = df[
                (df['teamName'] == opponent) &
                ((df['type'] == 'Pass') & (df['outcomeType'] == 'Unsuccessful') |
                 (df['type'].isin(['Dispossessed', 'LostDuel'])))
            ]
            num_ball_losses = len(ball_losses)
            if num_ball_losses > 100 and high_press_ratio < 0.15:
                high_press = True
                st.write(f"تم اكتشاف ضغط عالٍ لـ {team} بناءً على فقدان الكرة العالي ({num_ball_losses}).")
            st.write(f"فقدان الكرة من قبل {opponent} (لصالح {team}): {num_ball_losses}")
        else:
            num_ball_losses = 0
            st.warning(f"لم يتم العثور على فريق خصم لـ {team}.")
            return {}

        # تحديد مسافة الضغط تلقائيًا
        defensive_events = df[df['type'].isin(defs) & (df['teamName'] == team) & (df['x'] >= x_min) & (df['x'] <= x_max)]
        avg_distance = 8.0 if high_press else 5.0
        if len(defensive_events) >= 3:
            distances = []
            for i in range(len(defensive_events) - 1):
                dist = ((defensive_events.iloc[i]['x'] - defensive_events.iloc[i+1]['x'])**2 +
                        (defensive_events.iloc[i]['y'] - defensive_events.iloc[i+1]['y'])**2)**0.5
                if dist > 0:
                    distances.append(dist)
            if distances:
                avg_distance = np.percentile(distances, 75)
                avg_distance = min(max(avg_distance, 6.0), 12.0) if high_press else min(max(avg_distance, 4.0), 8.0)
        st.write(f"مسافة الضغط المحسوبة تلقائيًا لـ {team}: {avg_distance:.2f} متر")

        # إعدادات محاكاة "Pressure" الافتراضية
        pressure_time_window = 10 / 60  # 10 ثوانٍ
        pressure_distance_factor = 4.0
        max_pressure_events = 15
        pressure_weight = 0.7 if high_press_ratio > 0.3 else 0.5 if high_press_ratio < 0.15 else 0.6

        # محاكاة أحداث الضغط
        pressure_count = 0
        if 'Pressure' not in df['type'].unique():
            passes = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['teamName'] == opponent) & (df['x'] >= x_min) & (df['x'] <= x_max)]
            potential_pressure = df[
                (df['type'].isin(['Tackle', 'Challenge', 'Interception', 'BallRecovery'])) &
                (df['outcomeType'] == 'Successful') &
                (df['teamName'] == team) &
                (df['x'] >= x_min) & (df['x'] <= x_max) &
                (~df['qualifiers'].astype(str).str.contains('Error|Missed', na=False))
            ]
            if not potential_pressure.empty and not passes.empty:
                pressure_events = []
                for _, pressure_row in potential_pressure.iterrows():
                    relevant_passes = passes[
                        (abs(pressure_row['cumulative_mins'] - passes['cumulative_mins']) <= pressure_time_window) &
                        (((pressure_row['x'] - passes['x'])**2 + (pressure_row['y'] - passes['y'])**2)**0.5 <= avg_distance * pressure_distance_factor)
                    ]
                    for _, pass_row in relevant_passes.iterrows():
                        if pressure_count < max_pressure_events:
                            pressure_event = pressure_row.copy()
                            pressure_event['type'] = 'Pressure'
                            pressure_event['pressure_weight'] = pressure_weight
                            pressure_events.append(pressure_event)
                            pressure_count += 1
                if pressure_events:
                    pressure_df = pd.DataFrame(pressure_events)
                    df = pd.concat([df, pressure_df[df.columns.union(['pressure_weight'])]], ignore_index=True)
                    defs.append('Pressure')
                    st.write(f"أحداث 'Pressure' المحاكاة لـ {team}: {pressure_count}")
                else:
                    st.warning(f"لم يتم إنشاء أحداث 'Pressure' لـ {team} بسبب عدم وجود تمريرات مطابقة.")
            else:
                st.warning(f"لا توجد بيانات كافية لمحاكاة أحداث 'Pressure' لـ {team}.")

        # تصفية التمريرات الناجحة
        passes_allowed = df[
            (df['type'] == 'Pass') &
            (df['outcomeType'] == 'Successful') &
            (df['teamName'] == opponent) &
            (df['x'] >= x_min) &
            (df['x'] <= x_max) &
            (~df['qualifiers'].astype(str).str.contains('Corner|Freekick|Throwin|GoalKick|KickOff', na=False))
        ]
        if 'endX' in df.columns and 'endY' in df.columns:
            passes_allowed = passes_allowed.assign(
                pass_distance=np.sqrt((passes_allowed['endX'] - passes_allowed['x'])**2 + (passes_allowed['endY'] - passes_allowed['y'])**2)
            )
            passes_allowed = passes_allowed[passes_allowed['pass_distance'] >= min_pass_distance]
        num_passes = len(passes_allowed)

        # الكشف عن الخلل وتطبيق التصحيح التلقائي
        issues_detected = []
        auto_correction_applied = False
        sofascore_ppda = None

        # استخراج PPDA من Sofascore إذا تم توفير الرابط
        if sofascore_match_url:
            sofascore_ppda = fetch_sofascore_ppda(sofascore_match_url, team)
            if sofascore_ppda is not None:
                st.write(f"سيتم استخدام PPDA من Sofascore ({sofascore_ppda:.2f}) كمرجع لفريق {team} إذا لزم الأمر.")

        # خلل: عدد التمريرات منخفض
        if num_passes < 70:
            issues_detected.append(f"عدد التمريرات الناجحة منخفض جدًا لفريق {team} ({num_passes}).")
            # التصحيح: إعادة التصفية بدون حد أدنى للمسافة
            passes_allowed = df[
                (df['type'] == 'Pass') &
                (df['outcomeType'] == 'Successful') &
                (df['teamName'] == opponent) &
                (df['x'] >= x_min) &
                (df['x'] <= x_max) &
                (~df['qualifiers'].astype(str).str.contains('Corner|Freekick|Throwin|GoalKick|KickOff', na=False))
            ]
            num_passes = len(passes_allowed)
            auto_correction_applied = True
            st.warning(f"تم تخفيف تصفية التمريرات تلقائيًا لـ {team}. عدد التمريرات الجديد: {num_passes}")

        st.write(f"الفريق: {team}, التمريرات الناجحة المسموح بها (x من {x_min} إلى {x_max}): {num_passes}")

        # تصفية الأفعال الدفاعية
        defensive_actions = df[
            (df['type'].isin(defs)) &
            (df['teamName'] == team) &
            (df['x'] >= x_min) &
            (df['x'] <= x_max) &
            (~df['qualifiers'].astype(str).str.contains('Offensive|Tactical|Error', na=False))
        ]
        if 'BallRecovery' in defensive_actions['type'].unique():
            ball_recoveries = defensive_actions[defensive_actions['type'] == 'BallRecovery']
            filtered_recoveries = ball_recoveries
            if len(filtered_recoveries) > 0:
                defensive_actions = pd.concat([
                    defensive_actions[defensive_actions['type'] != 'BallRecovery'],
                    filtered_recoveries
                ], ignore_index=True)
            else:
                st.write(f"لا توجد أحداث 'BallRecovery' صالحة لـ {team} بعد التصفية.")

        # حساب عدد الأفعال الدفاعية
        if 'pressure_weight' in defensive_actions.columns:
            defensive_actions['weight'] = defensive_actions['pressure_weight'].fillna(1.0)
            num_defs = defensive_actions['weight'].sum()
        else:
            num_defs = len(defensive_actions)

        # خلل: عدد الأفعال الدفاعية منخفض أو مرتفع
        if num_defs < 3:
            issues_detected.append(f"عدد الأفعال الدفاعية منخفض جدًا لفريق {team} ({round(num_defs, 2)}).")
            # التصحيح: إعادة محاكاة "Pressure" بنطاق أوسع
            pressure_time_window = 15 / 60  # 15 ثانية
            pressure_distance_factor = 5.0
            max_pressure_events = 20
            pressure_weight = 0.95 if high_press else 0.7
            pressure_count = 0
            pressure_events = []
            if not potential_pressure.empty and not passes.empty:
                for _, pressure_row in potential_pressure.iterrows():
                    relevant_passes = passes[
                        (abs(pressure_row['cumulative_mins'] - passes['cumulative_mins']) <= pressure_time_window) &
                        (((pressure_row['x'] - passes['x'])**2 + (pressure_row['y'] - passes['y'])**2)**0.5 <= avg_distance * pressure_distance_factor)
                    ]
                    for _, pass_row in relevant_passes.iterrows():
                        if pressure_count < max_pressure_events:
                            pressure_event = pressure_row.copy()
                            pressure_event['type'] = 'Pressure'
                            pressure_event['pressure_weight'] = pressure_weight
                            pressure_events.append(pressure_event)
                            pressure_count += 1
                if pressure_events:
                    pressure_df = pd.DataFrame(pressure_events)
                    df = pd.concat([df, pressure_df[df.columns.union(['pressure_weight'])]], ignore_index=True)
                    if 'Pressure' not in defs:
                        defs.append('Pressure')
                    auto_correction_applied = True
                    st.warning(f"تمت إعادة محاكاة 'Pressure' بنطاق أوسع لـ {team}. عدد الأحداث الجديدة: {pressure_count}")
                    # إعادة حساب الأفعال الدفاعية
                    defensive_actions = df[
                        (df['type'].isin(defs)) &
                        (df['teamName'] == team) &
                        (df['x'] >= x_min) &
                        (df['x'] <= x_max) &
                        (~df['qualifiers'].astype(str).str.contains('Offensive|Tactical|Error', na=False))
                    ]
                    if 'pressure_weight' in defensive_actions.columns:
                        defensive_actions['weight'] = defensive_actions['pressure_weight'].fillna(1.0)
                        num_defs = defensive_actions['weight'].sum()
                    else:
                        num_defs = len(defensive_actions)

        if num_defs > 25:
            issues_detected.append(f"عدد الأفعال الدفاعية مرتفع جدًا لفريق {team} ({round(num_defs, 2)}).")
            # التصحيح: تقليل وزن "Pressure"
            if 'pressure_weight' in defensive_actions.columns:
                defensive_actions.loc[defensive_actions['type'] == 'Pressure', 'pressure_weight'] = 0.5
                defensive_actions['weight'] = defensive_actions['pressure_weight'].fillna(1.0)
                num_defs = defensive_actions['weight'].sum()
                auto_correction_applied = True
                st.warning(f"تم تقليل وزن أحداث 'Pressure' تلقائيًا لـ {team}. عدد الأفعال الجديد: {round(num_defs, 2)}")

        # خلل: أحداث "Pressure" قليلة مع ضغط عالٍ
        if high_press and pressure_count < 2:
            issues_detected.append(f"أحداث 'Pressure' قليلة جدًا لـ {team} ({pressure_count}) مع ضغط عالٍ.")
            # التصحيح: إعادة محاكاة بنطاق أوسع ووزن أعلى
            pressure_time_window = 15 / 60
            pressure_distance_factor = 5.0
            max_pressure_events = 20
            pressure_weight = 0.95
            pressure_count = 0
            pressure_events = []
            if not potential_pressure.empty and not passes.empty:
                for _, pressure_row in potential_pressure.iterrows():
                    relevant_passes = passes[
                        (abs(pressure_row['cumulative_mins'] - passes['cumulative_mins']) <= pressure_time_window) &
                        (((pressure_row['x'] - passes['x'])**2 + (pressure_row['y'] - passes['y'])**2)**0.5 <= avg_distance * pressure_distance_factor)
                    ]
                    for _, pass_row in relevant_passes.iterrows():
                        if pressure_count < max_pressure_events:
                            pressure_event = pressure_row.copy()
                            pressure_event['type'] = 'Pressure'
                            pressure_event['pressure_weight'] = pressure_weight
                            pressure_events.append(pressure_event)
                            pressure_count += 1
                if pressure_events:
                    pressure_df = pd.DataFrame(pressure_events)
                    df = pd.concat([df, pressure_df[df.columns.union(['pressure_weight'])]], ignore_index=True)
                    if 'Pressure' not in defs:
                        defs.append('Pressure')
                    auto_correction_applied = True
                    st.warning(f"تمت إعادة محاكاة 'Pressure' لضغط عالٍ لـ {team}. عدد الأحداث الجديدة: {pressure_count}")
                    # إعادة حساب الأفعال الدفاعية
                    defensive_actions = df[
                        (df['type'].isin(defs)) &
                        (df['teamName'] == team) &
                        (df['x'] >= x_min) &
                        (df['x'] <= x_max) &
                        (~df['qualifiers'].astype(str).str.contains('Offensive|Tactical|Error', na=False))
                    ]
                    if 'pressure_weight' in defensive_actions.columns:
                        defensive_actions['weight'] = defensive_actions['pressure_weight'].fillna(1.0)
                        num_defs = defensive_actions['weight'].sum()
                    else:
                        num_defs = len(defensive_actions)

        if num_defs < min_def_actions:
            st.warning(f"عدد الأفعال الدفاعية قليل جدًا لفريق {team} ({round(num_defs, 2)} < {min_def_actions}). سيتم حساب PPDA مع تحذير.")
        st.write(f"الفريق: {team}, الأفعال الدفاعية (x من {x_min} إلى {x_max}): {round(num_defs, 2)}")

        # حساب PPDA
        ppda = num_passes / num_defs if num_defs > 0 else float('inf')
        pressure_ratio = (num_defs / num_passes) * 100 if num_passes > 0 else None
        if ppda < 2 or ppda > 30:
            issues_detected.append(f"PPDA غير منطقي لفريق {team} ({round(ppda, 2)}).")

        st.write(f"PPDA الخام لفريق {team}: {round(ppda, 2)}")

        # معايرة ديناميكية
        calibration_factor = 1.0
        total_defs = df[df['type'].isin(defs) & (df['teamName'] == team)]['weight'].sum() if 'weight' in df.columns else len(df[df['type'].isin(defs) & (df['teamName'] == team)])
        region_def_ratio = num_defs / (total_defs + 1e-10)
        league_avg_ppda = 12.0

        if ppda < 2 or ppda > 30:
            if sofascore_ppda is not None:
                # استخدام PPDA من Sofascore مباشرة إذا كان متاحًا
                calibrated_ppda = sofascore_ppda
                calibration_factor = sofascore_ppda / ppda if ppda != 0 else 1.0
                auto_correction_applied = True
                st.warning(f"تم استخدام PPDA من Sofascore ({sofascore_ppda:.2f}) لفريق {team} بسبب قيمة غير منطقية.")
            else:
                # التصحيح الافتراضي إذا لم يتوفر PPDA من Sofascore
                if ppda > 30:
                    calibration_factor = 0.6 if num_defs < 5 else 0.75
                    if region_def_ratio < 0.2:
                        calibration_factor *= 0.85
                    if not high_press:
                        calibration_factor *= 0.8
                elif ppda < 2:
                    calibration_factor = 1.8 if high_press else 1.5
                calibrated_ppda = ppda * calibration_factor
                auto_correction_applied = True
                st.warning(f"تمت معايرة PPDA تلقائيًا لـ {team} بسبب قيمة غير منطقية.")
        else:
            if high_press_ratio > 0.3:
                calibration_factor = 0.8
            elif high_press_ratio < 0.15:
                calibration_factor = 1.3
            calibrated_ppda = ppda * calibration_factor
            # التحقق من التوافق مع PPDA من Sofascore إذا كان متاحًا
            if sofascore_ppda is not None and abs(calibrated_ppda - sofascore_ppda) / sofascore_ppda > 0.3:
                st.warning(f"PPDA المحسوب ({calibrated_ppda:.2f}) يختلف كثيرًا عن PPDA من Sofascore ({sofascore_ppda:.2f}) لفريق {team}.")
                # استخدام متوسط مرجح إذا كان الفرق كبيرًا
                calibrated_ppda = 0.7 * calibrated_ppda + 0.3 * sofascore_ppda
                calibration_factor = calibrated_ppda / ppda if ppda != 0 else 1.0
                auto_correction_applied = True
                st.write(f"تم تعديل PPDA إلى متوسط مرجح ({calibrated_ppda:.2f}) باستخدام Sofascore.")

        # ضمان النطاق المنطقي
        calibrated_ppda = min(max(calibrated_ppda, 3.0), 20.0)
        st.write(f"PPDA لفريق {team} معاير ({round(calibrated_ppda, 2)} بعد التصحيح). معامل المعايرة: {round(calibration_factor, 3)}.")

        # تسجيل الأخطاء والتصحيحات
        if issues_detected:
            st.warning(f"الأخطاء المكتشفة لـ {team}: {', '.join(issues_detected)}")
            if auto_correction_applied:
                st.write(f"تم تطبيق التصحيح التلقائي: {auto_correction_applied}")

        return {
            'Region': region,
            'Threshold_x_min': x_min,
            'Threshold_x_max': x_max,
            'Passes Allowed': num_passes,
            'Defensive Actions': round(num_defs, 2),
            'PPDA': round(calibrated_ppda, 2) if calibrated_ppda != float('inf') else None,
            'Sofascore_PPDA': round(sofascore_ppda, 2) if sofascore_ppda is not None else None,
            'Pressure Ratio (%)': round(pressure_ratio, 2) if pressure_ratio is not None else None,
            'Ball Losses Forced': num_ball_losses,
            'High Press': high_press,
            'Action Breakdown': {a: round(defensive_actions[defensive_actions['type'] == a]['weight'].sum(), 2) if 'weight' in defensive_actions.columns else int((defensive_actions['type'] == a).sum()) for a in defs}
        }

    except Exception as e:
        st.error(f"خطأ في حساب PPDA لفريق {team}: {str(e)}")
        return {}
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
            st.pyplot(fig)
            if pass_btn is not None and not pass_btn.empty:
                st.dataframe(pass_btn, hide_index=True)
            else:
                st.warning("لا توجد بيانات تمريرات للعرض.")
        except Exception as e:
            st.error(f"خطأ في إنشاء شبكة التمريرات: {str(e)}")

    elif an_tp == 'مناطق الهجوم':
        st.subheader('تحليل مناطق الهجوم')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
        attack_summary, fig_heatmap, fig_bar = attack_zones_analysis(fig, ax, hteamName, ateamName, hcol, acol, hteamID, ateamID)
        
        # عرض خريطة حرارية
        st.pyplot(fig_heatmap)
        
        # عرض الرسم البياني الشريطي
        st.pyplot(fig_bar)
        
        # عرض الجدول الإحصائي
        st.subheader('إحصائيات مناطق الهجوم')
        st.dataframe(attack_summary, hide_index=True)

    elif an_tp == reshape_arabic_text('Team Domination Zones'):
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
        st.pyplot(fig)

    elif an_tp == reshape_arabic_text('PPDA'):
        st.subheader(reshape_arabic_text('معدل الضغط (PPDA)'))
        st.write(reshape_arabic_text("PPDA: عدد التمريرات الناجحة التي يسمح بها الفريق مقابل كل فعل دفاعي في الثلث الدفاعي للخصم. القيمة الأقل تشير إلى ضغط دفاعي أقوى (عادة 5-15)."))

    # اختيار الفترة
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

    try:
        results = {}
        teams = st.session_state.df['teamName'].unique()
        for team in teams:
            st.write(f"حساب PPDA لـ {team}")
            results[team] = calculate_team_ppda(
                events_df=st.session_state.df,
                team=team,
                region='opponent_defensive_third',
                period=selected_period,
                pitch_units=105,
                min_def_actions=3,
                min_pass_distance=5.0
            )

        if not results or all(not res for res in results.values()):
            st.error("لا توجد نتائج PPDA متاحة.")
        else:
            # إنشاء إطار بيانات مع التحقق من القيم
            ppda_df = pd.DataFrame.from_dict(results, orient='index')
            required_columns = ['Passes Allowed', 'Defensive Actions', 'PPDA', 'Pressure Ratio (%)', 'Ball Losses Forced', 'High Press']
            for col in required_columns:
                if col not in ppda_df.columns:
                    ppda_df[col] = None
            ppda_df = ppda_df[required_columns].fillna('غير متاح')

            st.subheader(reshape_arabic_text('نتائج PPDA'))
            st.dataframe(
                ppda_df,
                use_container_width=True
            )

            st.subheader(reshape_arabic_text('تفاصيل الأفعال الدفاعية'))
            for team, result in results.items():
                if result and 'Action Breakdown' in result:
                    st.write(reshape_arabic_text(f"الفريق: {team}"))
                    action_df = pd.DataFrame.from_dict(result['Action Breakdown'], orient='index', columns=['Count'])
                    st.dataframe(action_df, use_container_width=True)
                else:
                    st.write(reshape_arabic_text(f"لا توجد تفاصيل أفعال دفاعية لـ {team}"))

            # رسم الرسم البياني
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
            ax.set_facecolor('#1a1a1a')
            colors = sns.color_palette("husl", len(ppda_df))
            ppda_values = pd.to_numeric(ppda_df['PPDA'], errors='coerce').fillna(0)
            bars = ax.bar(ppda_df.index, ppda_values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)

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
            ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5)
            ax.text(0, 12.5, reshape_arabic_text('متوسط PPDA في الدوري'), color='white')

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"خطأ في حساب PPDA: {str(e)}")
        st.write("يرجى التحقق من البيانات المحملة.")
