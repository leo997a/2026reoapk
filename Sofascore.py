# ppda_app.py
import streamlit as st

st.title("📊 حساب PPDA (Passes Per Defensive Action)")

st.write("""
أدخل البيانات المطلوبة من مصدر مثل Sofascore أو FBref.  
**ملاحظات**:
- PPDA = عدد التمريرات التي قام بها الخصم في نصف ملعبهم / عدد الأفعال الدفاعية (تدخلات + اعتراضات).
- تأكد من استخدام بيانات دقيقة من منطقة نصف الملعب الدفاعي.
""")

# اختيار المنطقة
area = st.selectbox("اختر المنطقة للتمريرات:", ["نصف الملعب الدفاعي", "الثلث الدفاعي"])

# مدخلات البيانات
passes = st.number_input(f"عدد التمريرات التي قام بها الخصم في {area}", min_value=0.0, step=1.0)
tackles = st.number_input("عدد التدخلات", min_value=0.0, step=1.0)
interceptions = st.number_input("عدد الاعتراضات", min_value=0.0, step=1.0)

# حساب الأفعال الدفاعية
actions = tackles + interceptions

# التحقق من المدخلات
if passes < 0 or tackles < 0 or interceptions < 0:
    st.error("❌ المدخلات لا يمكن أن تكون سالبة!")
elif actions == 0 and passes > 0:
    st.warning("⚠️ لا يمكن القسمة على صفر، تأكد من إدخال تدخلات أو اعتراضات.")
elif actions > 0:
    ppda = passes / actions
    st.success(f"✅ PPDA = {ppda:.2f}")
    st.info(f"تفسير: قيمة PPDA منخفضة (مثل <10) تشير إلى ضغط دفاعي عالي، بينما القيم المرتفعة (>15) تشير إلى ضغط أقل.")

# إرشادات إضافية
st.write("""
### كيفية استخراج البيانات:
- **Sofascore**: ابحث عن "Passes in opponents' half" للتمريرات في نصف الملعب الدفاعي، و"Tackles" و"Interceptions" للأفعال الدفاعية.
- **FBref**: استخدم إحصائيات "Possession" و"Defensive Actions" لاستخراج البيانات.
- تأكد من مطابقة المنطقة المختارة (نصف الملعب أو الثلث الدفاعي) مع البيانات.
""")
