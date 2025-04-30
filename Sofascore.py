import streamlit as st
from bs4 import BeautifulSoup
import re

st.title("📄 حساب PPDA من ملف HTML محفوظ")

uploaded_file = st.file_uploader("قم برفع ملف HTML لصفحة المباراة من Sofascore", type="html")

if uploaded_file is not None:
    soup = BeautifulSoup(uploaded_file, "html.parser")
    text = soup.get_text()

    # استخدم تعبيرات مناسبة حسب شكل النص داخل الصفحة
    passes = re.search(r"Passes in defensive third\s*(\d+)", text)
    actions = re.search(r"Defensive actions\s*(\d+)", text)

    if passes and actions:
        passes_val = int(passes.group(1))
        actions_val = int(actions.group(1))

        if actions_val == 0:
            st.warning("⚠️ لا يمكن القسمة على صفر.")
        else:
            ppda = passes_val / actions_val
            st.success(f"✅ PPDA = {ppda:.2f}")
    else:
        st.error("❌ لم يتم العثور على البيانات المطلوبة في الملف. تحقق من الكلمات المفتاحية.")
else:
    st.info("⬆️ الرجاء رفع ملف HTML أولاً.")
