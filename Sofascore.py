# ppda_app.py
import streamlit as st

st.title("📊 حساب PPDA (Passes Per Defensive Action)")

st.write("أدخل البيانات المطلوبة كما هي موجودة في Sofascore أو مصدر آخر.")

passes = st.number_input("عدد التمريرات التي قام بها الخصم في الثلث الدفاعي", min_value=0)
actions = st.number_input("عدد الأفعال الدفاعية (اعتراضات، ضغط، تدخل...)", min_value=0)

if actions > 0:
    ppda = passes / actions
    st.success(f"✅ PPDA = {ppda:.2f}")
elif passes > 0:
    st.warning("⚠️ لا يمكن القسمة على صفر، تأكد من إدخال عدد الأفعال الدفاعية.")
