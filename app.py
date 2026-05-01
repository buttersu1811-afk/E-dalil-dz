import chromadb  # مكتبة قاعدة البيانات المتجهية لتخزين النصوص القانونية
from sentence_transformers import SentenceTransformer  # مكتبة تحويل النصوص إلى متجهات (أرقام)

# 1. الاتصال بقاعدة البيانات
client = chromadb.PersistentClient(path="legal_db")  # إنشاء اتصال بقاعدة بيانات محلية باسم legal_db
collection = client.get_collection(name="algerian_law")  # الحصول على مجموعة البيانات المسماة algerian_law

# 2. تحميل نموذج الفهم (نفس النموذج المستخدم في التخزين)
print("⏳ جاري تحميل نموذج الذكاء الاصطناعي...")  # رسالة للمستخدم بجاري التحميل
model = SentenceTransformer('intfloat/multilingual-e5-small')  # تحميل نموذج multilingual-e5-small للغة العربية والإنجليزية


def search(query):  # دالة البحث: تأخذ سؤال المستخدم
    print(f"\n🔍 جاري البحث عن: '{query}'...")  # طباعة رسالة البحث

    # تحويل السؤال إلى أرقام (متجه)
    query_vector = model.encode("query: " + query).tolist()  # إضافة بادئة query: لنموذج e5، ثم تحويل إلى قائمة

    # البحث عن أقرب 3 فقرات
    results = collection.query(  # استعلام عن قاعدة البيانات
        query_embeddings=[query_vector],  # المتجه الذي نريد البحث عنه
        n_results=3  # عدد النتائج المطلوبة
    )

    documents = results['documents'][0]  # استخراج النصوص المتطابقة (أول مصفوفة)
    ids = results['ids'][0]  # استخراج المعرفات الفريدة لكل نتيجة

    if not documents:  # إذا لم يتم العثور على نتائج
        print("❌ لم يتم العثور على نتائج.")  # طباعة رسالة فشل
        return  # الخروج من الدالة

    print(f"\n✅ وجدنا {len(documents)} نتائج ذات صلة:\n")  # طباعة عدد النتائج
    for i in range(len(documents)):  # التكرار على كل نتيجة
        print(f"--- النتيجة {i + 1} (المصدر: {ids[i]}) ---")  # طباعة رقم النتيجة والمصدر
        print(documents[i])  # طباعة النص القانوني نفسه
        print("-" * 40)  # خط فاصل


# --- واجهة التجربة ---
if __name__ == "__main__":  # إذا تم تشغيل الملف مباشرة (وليس استيراده)
    print("\n⚖️  مرحباً بك في نظام البحث القانوني الجزائري ⚖️")  # رسالة ترحيب
    while True:  # حلقة لا نهائية
        q = input("\nأدخل سؤالك القانوني (أو exit للخروج): ")  # طلب إدخال من المستخدم
        if q.lower() == "exit":  # إذا كان الإدخال هو exit
            break  # الخروج من الحلقة
        search(q)  # استدعاء دالة البحث مع السؤال