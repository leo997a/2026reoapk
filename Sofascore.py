from bs4 import BeautifulSoup

def extract_ppda_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # هنا تحتاج لتعديل المسارات بناءً على هيكل الصفحة الفعلي
    # سنبحث عن التمريرات في الثلث الدفاعي للفريق (مثلاً)
    # ومجمل الأفعال الدفاعية مثل interceptions, tackles, pressures
    
    # مثال: نبحث عن div أو span يحتوي على النصوص
    all_text = soup.get_text()

    # ثم نستخرج منها الأرقام (ستحتاج لضبط هذا حسب اللغة والمصدر)
    import re

    try:
        passes = int(re.search(r"Passes in defensive third\s+(\d+)", all_text).group(1))
        actions = int(re.search(r"Defensive actions\s+(\d+)", all_text).group(1))
    except AttributeError:
        print("لم يتم العثور على البيانات المطلوبة داخل الملف.")
        return

    if actions == 0:
        print("⚠️ لا يمكن القسمة على صفر.")
        return

    ppda = passes / actions
    print(f"✅ PPDA = {ppda:.2f}")

# مثال: ضع مسار ملف HTML هنا
extract_ppda_from_html("match.html")
