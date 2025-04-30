import requests
from bs4 import BeautifulSoup

def extract_ppda(sofa_url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(sofa_url, headers=headers)
    if response.status_code != 200:
        print("فشل في تحميل الصفحة.")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # هذا يعتمد على هيكل الصفحة - ستحتاج إلى تحديثه حسب البيانات الدقيقة في صفحة Sofascore
    # هذا كود مبدأي يبحث عن إحصائيات التمريرات والأفعال الدفاعية
    
    stats = soup.find_all('div', class_='sc-...')
    # هنا تستخرج الإحصائيات من الصفحة - نحتاج لتحديد المسارات الصحيحة من الصفحة يدويًا أو باستخدام أدوات devtools

    # بديل أفضل: إذا كنت تملك بيانات مثل:
    passes_in_defensive_third = 85  # مثال
    defensive_actions = 10          # مثال

    ppda = passes_in_defensive_third / defensive_actions if defensive_actions > 0 else None
    print(f"PPDA = {ppda}")

# مثال:
match_url = "https://www.sofascore.com/match/..."
extract_ppda(match_url)
