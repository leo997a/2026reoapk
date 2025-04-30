import requests
from bs4 import BeautifulSoup

# تحميل الصفحة مباشرة باستخدام requests (إذا كنت تفضل عدم تحميل HTML يدويًا)
url = "https://www.sofascore.com/ar/football/match/wqf1-wqf2/RsUH#id:13513415"
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
if response.status_code != 200:
    print("❌ فشل تحميل الصفحة")
    exit()

# تحليل HTML
soup = BeautifulSoup(response.content, 'html.parser')

# ابحث عن التمريرات والأفعال الدفاعية داخل الصفحة
# ستحتاج هنا إلى تحديد العناصر المناسبة بناءً على هيكل HTML الخاص بالصفحة
# كمثال، نحن نبحث عن العناصر التي تحتوي على إحصائيات التمريرات والأفعال الدفاعية

# مثال للبحث عن النصوص (يجب تعديل الـ CSS selectors أو الـ class names حسب هيكل الصفحة)
passes_in_defensive_third = soup.find('div', class_='pass-class')  # استبدل بالـ class الفعلي
defensive_actions = soup.find('div', class_='defensive-actions-class')  # استبدل بالـ class الفعلي

if passes_in_defensive_third and defensive_actions:
    print(f"التمريرات في الثلث الدفاعي: {passes_in_defensive_third.text.strip()}")
    print(f"الأفعال الدفاعية: {defensive_actions.text.strip()}")
else:
    print("❌ لم يتم العثور على الإحصائيات المطلوبة.")
