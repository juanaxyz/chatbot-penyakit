import json
from server import (
    search_disease_by_location,
    get_article_by_disease,
    get_disease_urgency,
    get_pathogen_by_disease,
    get_disease_references,
    search_faq,
    app,
)
from fastapi.testclient import TestClient

# 1. search_disease_by_location
res1 = search_disease_by_location.invoke({"location": "dada"})
data1 = json.loads(res1)
assert isinstance(data1, list) and len(data1) > 0
print("1. search_disease_by_location: OK (found", len(data1), ")")

# 2. get_article_by_disease
res2 = get_article_by_disease.invoke({"disease_name": "Bronkitis"})
data2 = json.loads(res2)
assert "konten" in data2 and len(data2["konten"]) > 50
print("2. get_article_by_disease: OK (chars:", len(data2["konten"]), ")")

# 3. get_disease_urgency
res3 = get_disease_urgency.invoke({"disease_name": "Pneumotoraks"})
data3 = json.loads(res3)
assert data3.get("urgensi") == "darurat"
print("3. get_disease_urgency: OK (urgensi:", data3.get("urgensi"), ")")

# 4. get_pathogen_by_disease
res4 = get_pathogen_by_disease.invoke({"disease_name": "COVID-19"})
data4 = json.loads(res4)
assert isinstance(data4, list) and len(data4) > 0 and data4[0]["jenis"] == "virus"
print("4. get_pathogen_by_disease: OK (patogen:", data4[0]["patogen"], ")")

# 5. get_disease_references
res5 = get_disease_references.invoke({"disease_name": "Tuberkulosis"})
data5 = json.loads(res5)
assert isinstance(data5, list) and len(data5) > 0
print("5. get_disease_references: OK (urls:", len(data5), ")")

# 6. search_faq
res6 = search_faq.invoke({"keyword": "body map"})
data6 = json.loads(res6)
assert isinstance(data6, list) and len(data6) > 0
print("6. search_faq: OK (faqs:", len(data6), ")")

# 7. Test endpoint
client = TestClient(app)
assert client.get("/health").status_code == 200
html_res = client.get("/")
assert html_res.status_code == 200
print("7. GET /health & GET / HTML endpoint: OK")

print("\nSemua 6 tool database terverifikasi 100% lulus uji!")
