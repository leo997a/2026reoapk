import streamlit as st
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="PPDA Extractor", layout="centered")
st.title("📊 حساب PPDA من ملف HTML محفوظ")

st.write("🔽 قم برفع ملف HTML المحفوظ من صفحة مباراة Sofascore:")

uploaded_file = st.file_uploader("اختر ملف HTML", type="html")

if uploaded_file is not None:
    soup = BeautifulSoup(uploaded_file, "html.parser")
    page_text = soup.get_text()

    # تعديل عبارات البحث حسب النصوص الفعلية التي تجدها في الملف
    passes_match = re.search(r"التمريرات في الثلث الدفاعي\s*(\d+)", page_text)  # مثال للغة العربية
    actions_match = re.search(r"الأفعال الدفاعية\s*(\d+)", page_text)  # مثال للغة العربية

    if passes_match and actions_match:
        passes = int(passes_match.group(1))
        actions = int(actions_match.group(1))

        if actions == 0:
            st.warning("⚠️ لا يمكن حساب PPDA لأن عدد الأفعال الدفاعية = 0")
        else:
            ppda = passes / actions
            st.success(f"✅ PPDA = {ppda:.2f}")
    else:
        st.error("❌ لم يتم العثور على البيانات المطلوبة. تأكد من أن الملف يحتوي على 'التمريرات في الثلث الدفاعي' و 'الأفعال الدفاعية'.")
else:
    st.info("⬆️ الرجاء رفع ملف HTML أولاً.")
