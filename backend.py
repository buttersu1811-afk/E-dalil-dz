import os  # مكتبة للتعامل مع نظام الملفات (إنشاء، حذف، مسارات)
import json  # مكتبة للتعامل مع ملفات JSON (تخزين البيانات)
import tempfile  # مكتبة لإنشاء ملفات مؤقتة
import time  # مكتبة للتعامل مع الوقت (تأخير، تواريخ)
import subprocess  # مكتبة لتشغيل أوامر النظام
import sys  # مكتبة للتعامل مع النظام والمتغيرات
from datetime import datetime, timedelta  # استيراد دوال التاريخ والوقت

import chromadb  # قاعدة البيانات المتجهية
import pdfplumber  # استخراج النص من ملفات PDF
import arabic_reshaper  # إعادة تشكيل الحروف العربية المتقطعة
from bidi.algorithm import get_display  # معالجة اتجاه النص العربي
from chromadb.errors import NotFoundError  # خطأ عند عدم وجود مجموعة بيانات
from sentence_transformers import SentenceTransformer  # تحويل النصوص لمتجهات
import ollama  # مكتبة للتعامل مع نماذج Ollama المحلية
from flask import Flask, request, jsonify, send_from_directory, send_file  # إنشاء خادم ويب
from flask_cors import CORS  # السماح بمشاركة الموارد بين الخوادم
from dotenv import load_dotenv  # تحميل المتغيرات البيئية من ملف .env
import easyocr  # مكتبة التعرف على النصوص في الصور
import fitz  # PyMuPDF - مكتبة متقدمة للتعامل مع PDF

app = Flask(__name__)  # إنشاء تطبيق Flask
CORS(app)  # تفعيل CORS للسماح بالطلبات من الواجهة الأمامية
load_dotenv()  # تحميل المتغيرات من ملف .env

# ========== تحميل النماذج ==========
print("⏳ جاري تحميل نموذج التضمين (Embedding)...")  # رسالة تحميل
model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')  # تحميل نموذج التضمين

print("⏳ جاري تهيئة EasyOCR للغة العربية...")  # رسالة تهيئة EasyOCR
try:  # محاولة تهيئة EasyOCR
    easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)  # إنشاء قارئ OCR للعربية والإنجليزية (بدون GPU)
except Exception as e:  # في حالة حدوث خطأ
    print(f"⚠️ خطأ في تهيئة EasyOCR: {e}")  # طباعة الخطأ
    easyocr_reader = None  # تعيين القيمة None لاستخدامها لاحقاً

# ========== الاتصال بقاعدة البيانات ==========
client_db = chromadb.PersistentClient(path="legal_db")  # اتصال بقاعدة البيانات
try:  # محاولة الحصول على المجموعة
    collection = client_db.get_collection(name="algerian_law")  # الحصول على مجموعة algerian_law
except NotFoundError:  # إذا لم تكن موجودة
    print("⚠️ المجموعة غير موجودة، سيتم إنشاؤها...")  # رسالة تحذير
    collection = client_db.create_collection(name="algerian_law")  # إنشاء مجموعة جديدة

# ========== سجل الملفات المضافة ==========
HISTORY_FILE = "processed_history.json"  # اسم ملف السجل

def load_history():  # دالة لتحميل سجل الملفات
    if os.path.exists(HISTORY_FILE):  # إذا كان الملف موجوداً
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:  # فتح الملف للقراءة
            return set(json.load(f))  # تحويل القائمة إلى مجموعة (set) وإرجاعها
    return set()  # إرجاع مجموعة فارغة إذا لم يكن الملف موجوداً

def save_history(history):  # دالة لحفظ السجل
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:  # فتح الملف للكتابة
        json.dump(list(history), f, ensure_ascii=False, indent=2)  # تحويل المجموعة إلى قائمة وحفظها

# ========== دوال تنظيف النص العربي ==========
def clean_arabic_text(text):  # دالة لتنظيف النص العربي
    if not text:  # إذا كان النص فارغاً
        return ""  # إرجاع نص فارغ
    try:  # محاولة تنظيف النص
        reshaped = arabic_reshaper.reshape(text)  # إعادة تشكيل الحروف العربية
        bidi_text = get_display(reshaped)  # معالجة اتجاه النص
        return bidi_text  # إرجاع النص المعالج
    except Exception as e:  # في حالة حدوث خطأ
        print(f"⚠️ خطأ في تنظيف النص: {e}")  # طباعة الخطأ
        return text  # إرجاع النص الأصلي بدون تعديل

def has_substantial_text(text):  # دالة للتحقق من وجود نص كافٍ
    return text and len(text.strip()) > 50  # إرجاع True إذا النص موجود وطوله أكبر من 50 حرفاً

# ========== استخراج النص من PDF/صور ==========
def extract_text_with_fallback(file_path, file_extension=None):  # دالة استخراج النص مع خيارات احتياطية
    full_text = ""  # متغير لتخزين النص الكامل
    method_used = None  # متغير لتخزين الطريقة المستخدمة

    if file_extension == '.pdf' or (file_extension is None and file_path.lower().endswith('.pdf')):  # إذا كان الملف PDF
        try:  # محاولة استخراج النص باستخدام pdfplumber
            with pdfplumber.open(file_path) as pdf:  # فتح ملف PDF
                for page in pdf.pages:  # التكرار على كل صفحة
                    text = page.extract_text()  # استخراج النص من الصفحة
                    if text:  # إذا كان هناك نص
                        full_text += text + "\n"  # إضافة النص إلى المتغير
            if has_substantial_text(full_text):  # إذا كان النص كافياً
                method_used = "pdfplumber (نصوص رقمية)"  # تحديد الطريقة المستخدمة
                return full_text, method_used  # إرجاع النص والطريقة
        except Exception as e:  # في حالة فشل pdfplumber
            print(f"⚠️ فشل pdfplumber: {e}")  # طباعة الخطأ

        if easyocr_reader:  # إذا كان EasyOCR متاحاً
            try:  # محاولة استخدام EasyOCR
                doc = fitz.open(file_path)  # فتح PDF باستخدام PyMuPDF
                ocr_text = ""  # متغير للنص المستخرج بـ OCR
                for page_num in range(len(doc)):  # التكرار على كل صفحة
                    pix = doc.load_page(page_num).get_pixmap(dpi=150)  # تحويل الصفحة إلى صورة بدقة 150 DPI
                    img_path = f"temp_page_{page_num}.png"  # اسم ملف مؤقت للصورة
                    pix.save(img_path)  # حفظ الصورة
                    result = easyocr_reader.readtext(img_path, detail=0, paragraph=True)  # استخراج النص من الصورة
                    if result:  # إذا كان هناك نص
                        ocr_text += " ".join(result) + "\n"  # إضافة النص
                    os.remove(img_path)  # حذف الصورة المؤقتة
                doc.close()  # إغلاق ملف PDF
                if has_substantial_text(ocr_text):  # إذا كان النص كافياً
                    return ocr_text, "EasyOCR (PDF ممسوح)"  # إرجاع النص والطريقة
            except Exception as e:  # في حالة فشل EasyOCR
                print(f"⚠️ فشل EasyOCR: {e}")  # طباعة الخطأ

    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:  # إذا كان الملف صورة
        if easyocr_reader:  # إذا كان EasyOCR متاحاً
            result = easyocr_reader.readtext(file_path, detail=0, paragraph=True)  # استخراج النص من الصورة
            if result:  # إذا كان هناك نص
                return " ".join(result), "EasyOCR (صورة)"  # إرجاع النص والطريقة

    return "", None  # إرجاع نص فارغ إذا فشل كل شيء

def chunk_text(text, chunk_size=500):  # دالة لتقطيع النص إلى أجزاء
    words = text.split()  # تقسيم النص إلى كلمات
    chunks = []  # قائمة لتخزين الأجزاء
    buffer = []  # مخزن مؤقت
    current_len = 0  # الطول الحالي
    for word in words:  # التكرار على كل كلمة
        buffer.append(word)  # إضافة الكلمة للمخزن
        current_len += len(word) + 1  # زيادة الطول
        if current_len >= chunk_size:  # إذا وصلنا للحجم المطلوب
            chunks.append(" ".join(buffer))  # تحويل المخزن لنص وإضافته للأجزاء
            buffer = []  # إفراغ المخزن
            current_len = 0  # إعادة تعيين الطول
    if buffer:  # إذا بقي شيء في المخزن
        chunks.append(" ".join(buffer))  # إضافته للأجزاء
    return chunks  # إرجاع قائمة الأجزاء

def add_file_to_library(file_path, original_filename=None):  # دالة لإضافة ملف للمكتبة
    display_name = original_filename or os.path.basename(file_path)  # اسم الملف للعرض
    history = load_history()  # تحميل سجل الملفات
    if display_name in history:  # إذا كان الملف موجوداً مسبقاً
        return f"⚠️ الملف '{display_name}' تمت إضافته مسبقاً."  # رسالة تحذير

    full_text, method = extract_text_with_fallback(file_path, os.path.splitext(file_path)[1].lower())  # استخراج النص
    if not full_text:  # إذا لم يتم استخراج نص
        return "❌ تعذر استخراج النص."  # رسالة فشل

    clean_text = clean_arabic_text(full_text)  # تنظيف النص العربي
    chunks = chunk_text(clean_text)  # تقطيع النص

    embeddings = [model_embedding.encode("passage: " + c).tolist() for c in chunks]  # تحويل كل جزء لمتجه
    ids = [f"{display_name}_{idx}" for idx in range(len(chunks))]  # إنشاء معرفات فريدة لكل جزء
    metadatas = [{"source": display_name} for _ in chunks]  # بيانات وصفية لكل جزء

    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)  # إضافة البيانات للمجموعة
    history.add(display_name)  # إضافة الملف للسجل
    save_history(history)  # حفظ السجل
    return f"✅ تمت إضافة '{display_name}' بنجاح عبر {method}."  # رسالة نجاح

def initial_scan_and_build():  # دالة لمسح المجلدات وإضافة الملفات تلقائياً
    if collection.count() > 0:  # إذا كانت المجموعة تحتوي بيانات بالفعل
        return  # الخروج دون فعل شيء
    folders = ["data/01--- قوانين وزارة التجارة", "data/التجارة الالكترونية", "data/قوانين السجل التجاري", "data/contrats_exemples", "data/كتب تجارية"]  # المجلدات المطلوب مسحها
    for folder in folders:  # التكرار على كل مجلد
        if os.path.exists(folder):  # إذا كان المجلد موجوداً
            for root, _, files in os.walk(folder):  # التكرار على كل ملف في المجلد
                for file in files:  # التكرار على الملفات
                    if file.lower().endswith('.pdf'):  # إذا كان ملف PDF
                        print(add_file_to_library(os.path.join(root, file)))  # إضافة الملف للمكتبة وطباعة النتيجة

# ========== دالة الإجابة باستخدام Ollama ==========
def ask_lawyer(query):  # دالة للإجابة على الأسئلة
    try:  # محاولة
        query_embedding = model_embedding.encode([query]).tolist()  # تحويل السؤال لمتجه
        results = collection.query(query_embeddings=query_embedding, n_results=3)  # البحث عن أقرب 3 نتائج

        context = "\n".join(results['documents'][0]) if results['documents'] else "لا يوجد سياق قانوني متاح."  # بناء السياق من النتائج

        full_prompt = f"""أنت مستشار قانوني جزائري محترف.  # بناء النص الموجه للنموذج
مهمتك: تقديم إجابات قانونية دقيقة ومنظمة بناءً فقط على النصوص القانونية المرفقة.

قواعد التنسيق الإلزامية:
- استخدم عناوين رئيسية على شكل: 1-العنوان (استخدم الأرقام)
- استخدم عناوين فرعية على شكل: أ-العنوان(استخدم الحرف)
- افصل بين كل عنوان و عنوان بسطر فارغ.
- استعمل خط كبير للعنوانين و خط رقيق للاجابة 
- لا تنسخ النص حرفياً من المصادر، بل أعد صياغته بلغة قانونية واضحة ومختصرة.
- لا تذكر عبارات مثل "بناءً على النصوص أعلاه" أو "وفقاً للمصادر". ابدأ الإجابة مباشرة.

النصوص القانونية:
{context}

سؤال المستخدم:
{query}

الإجابة (باللغة العربية):"""
        response = ollama.chat(model='aya', messages=[{'role': 'user', 'content': full_prompt}])  # إرسال الطلب لنموذج aya
        return {"answer": response['message']['content']}  # إرجاع الإجابة
    except Exception as e:  # في حالة حدوث خطأ
        print(f"❌ خطأ في Ollama: {e}")  # طباعة الخطأ
        return {"answer": "حدث خطأ أثناء محاولة معالجة السؤال محلياً. تأكد من تشغيل برنامج Ollama."}  # رسالة خطأ

# ========== مولد العقود ==========
def generate_contract(contract_type, parties, subject, duration, amount):  # دالة لتوليد العقود
    prompt = f"""أنت مستشار قانوني جزائري متخصص في صياغة العقود.  # النص الموجه للنموذج
بناءً على النصوص القانونية الجزائرية (قانون التجارة، قانون الصفقات العمومية، القانون المدني)، قم بإنشاء عقد كامل من نوع "{contract_type}" يتضمن المواد التالية على الأقل:
- تعريف الأطراف
- موضوع العقد
- المدة: {duration}
- المبلغ: {amount}
- التزامات الطرفين
- شروط الدفع
- الجزاءات والغرامات التأخيرية
- الضمانات
- تسوية النزاعات (التحكيم أو المحاكم الجزائرية)
- أحكام عامة (القوة القاهرة، اللغة، عدد النسخ)

أطراف العقد:
{parties}

موضوع العقد:
{subject}

اكتب العقد بلغة قانونية واضحة، مرقماً المواد (مادة 1، مادة 2...)، منسقاً بأسطر فارغة بين المواد. لا تذكر أي جمل تمهيدية مثل "بناءً على طلبك". ابدأ مباشرة بنص العقد.
"""
    response = ollama.chat(model='aya', messages=[{'role': 'user', 'content': prompt}])  # إرسال الطلب
    return response['message']['content']  # إرجاع نص العقد

@app.route('/generate_contract', methods=['POST'])  # نقطة نهاية لتوليد العقود
def api_generate_contract():  # دالة API
    data = request.get_json()  # استلام البيانات من الطلب
    contract_type = data.get('type')  # نوع العقد
    parties = data.get('parties')  # الأطراف
    subject = data.get('subject')  # الموضوع
    duration = data.get('duration')  # المدة
    amount = data.get('amount')  # المبلغ
    if not all([contract_type, parties, subject]):  # إذا كانت الحقول المطلوبة فارغة
        return jsonify({"error": "يرجى ملء الحقول المطلوبة"}), 400  # رسالة خطأ
    contract_text = generate_contract(contract_type, parties, subject, duration, amount)  # توليد العقد
    return jsonify({"contract": contract_text})  # إرجاع العقد

@app.route('/download_contract_pdf', methods=['POST'])  # نقطة نهاية لتحميل العقد كـ PDF
def download_contract_pdf():  # دالة API
    data = request.get_json()  # استلام البيانات
    contract_text = data.get('contract')  # نص العقد
    if not contract_text:  # إذا لم يكن هناك نص
        return jsonify({"error": "لا يوجد نص عقد"}), 400  # رسالة خطأ
    try:  # محاولة استيراد fpdf
        from fpdf import FPDF  # استيراد مكتبة FPDF
    except ImportError:  # إذا لم تكن مثبتة
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])  # تثبيتها تلقائياً
        from fpdf import FPDF  # استيرادها بعد التثبيت
    pdf = FPDF()  # إنشاء كائن PDF جديد
    pdf.add_page()  # إضافة صفحة
    pdf.set_font('Helvetica', size=12)  # تعيين الخط
    for line in contract_text.split('\n'):  # التكرار على كل سطر
        pdf.multi_cell(0, 10, line)  # إضافة السطر للـ PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')  # إنشاء ملف مؤقت
    pdf.output(temp_file.name)  # حفظ PDF في الملف المؤقت
    return send_file(temp_file.name, as_attachment=True, download_name='contrat_genere.pdf')  # إرسال الملف للتحميل

# ========== محلل المستندات ==========
@app.route('/analyze_document', methods=['POST'])  # نقطة نهاية لتحليل المستندات
def analyze_document():  # دالة API
    if 'file' not in request.files:  # إذا لم يتم إرسال ملف
        return jsonify({"error": "لا يوجد ملف"}), 400  # رسالة خطأ
    file = request.files['file']  # استلام الملف
    ext = os.path.splitext(file.filename)[1].lower()  # استخراج امتداد الملف
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:  # إنشاء ملف مؤقت
        file.save(tmp.name)  # حفظ الملف المؤقت
        text, method = extract_text_with_fallback(tmp.name, ext)  # استخراج النص منه
    os.unlink(tmp.name)  # حذف الملف المؤقت
    if not text:  # إذا لم يتم استخراج نص
        return jsonify({"error": "تعذر استخراج النص من الملف"}), 400  # رسالة خطأ
    prompt = f"""أنت خبير قانوني جزائري. قم بتحليل النص التالي وأخرج:  # النص الموجه لتحليل المستند
1. ملخص (3-5 جمل)
2. نقاط الخطر القانونية (Clauses à risque) - إن وجدت، وإلا اذكر "لا توجد نقاط خطر واضحة"
3. توصيات عملية للمستخدم

النص:
{text[:4000]}

أجب بالتنسيق التالي:
**الملخص:**
...
**نقاط الخطر:**
- ...
**التوصيات:**
- ...
"""
    response = ollama.chat(model='aya', messages=[{'role': 'user', 'content': prompt}])  # إرسال الطلب للنموذج
    return jsonify({"analysis": response['message']['content']})  # إرجاع التحليل

# ========== حاسبة المواعيد القانونية ==========
LEGAL_DEADLINES = {  # قاموس بالمواعيد القانونية بالأيام
    "تقادم دعوى مدنية": 15,  # 15 سنة للتقادم المدني
    "تقادم دعوى تجارية": 10,  # 10 سنوات للتقادم التجاري
    "الطعن في صفقة عمومية (بعد التبليغ)": 60,  # 60 يوماً للطعن في الصفقات العمومية
    "الطعن في قرار إداري": 30,  # 30 يوماً للطعن في القرارات الإدارية
    "إنهاء عقد عمل (إشعار مسبق)": 30,  # 30 يوماً لإشعار إنهاء العقد
}

@app.route('/calculate_deadlines', methods=['POST'])  # نقطة نهاية لحساب المواعيد
def calculate_deadlines():  # دالة API
    data = request.get_json()  # استلام البيانات
    action = data.get('action')  # نوع الإجراء
    start_date_str = data.get('start_date')  # تاريخ البدء
    if not action or not start_date_str:  # إذا كانت البيانات ناقصة
        return jsonify({"error": "يرجى تحديد الإجراء والتاريخ"}), 400  # رسالة خطأ
    if action not in LEGAL_DEADLINES:  # إذا كان الإجراء غير معروف
        return jsonify({"error": "نوع الإجراء غير معروف"}), 400  # رسالة خطأ
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')  # تحويل التاريخ من نص لكائن datetime
    days = LEGAL_DEADLINES[action]  # عدد الأيام المطلوبة
    delta = timedelta(days=days)  # إنشاء كائن timedelta
    end_date = start_date + delta  # حساب التاريخ النهائي
    return jsonify({  # إرجاع النتيجة
        "action": action,  # الإجراء
        "start_date": start_date_str,  # تاريخ البدء
        "deadline_date": end_date.strftime('%Y-%m-%d'),  # التاريخ النهائي
        "days_remaining": max(0, (end_date - datetime.now()).days),  # الأيام المتبقية
        "legal_basis": "المادة المرجعية حسب القانون الجزائري"  # الأساس القانوني
    })

# ========== مسارات API ==========
@app.route('/')  # نقطة نهاية الصفحة الرئيسية
def serve_index():  # دالة API
    return send_from_directory('.', 'index.html')  # إرسال ملف index.html

@app.route('/ask', methods=['POST'])  # نقطة نهاية الأسئلة
def ask():  # دالة API
    data = request.get_json()  # استلام البيانات
    query = data.get('query', '')  # السؤال
    if not query:  # إذا كان السؤال فارغاً
        return jsonify({"error": "الرجاء إدخال سؤال"}), 400  # رسالة خطأ
    return jsonify(ask_lawyer(query))  # إرجاع إجابة السؤال

@app.route('/upload', methods=['POST'])  # نقطة نهاية رفع الملفات
def upload():  # دالة API
    if 'file' not in request.files:  # إذا لم يتم إرسال ملف
        return jsonify({"error": "لا يوجد ملف"}), 400  # رسالة خطأ
    file = request.files['file']  # استلام الملف
    file_extension = os.path.splitext(file.filename)[1].lower()  # استخراج الامتداد
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:  # إنشاء ملف مؤقت
        file.save(tmp.name)  # حفظ الملف
        result = add_file_to_library(tmp.name, file.filename)  # إضافة الملف للمكتبة
    os.unlink(tmp.name)  # حذف الملف المؤقت
    return jsonify({"message": result})  # إرجاع النتيجة

@app.route('/stats', methods=['GET'])  # نقطة نهاية الإحصائيات
def stats():  # دالة API
    return jsonify({"chunks_count": collection.count()})  # إرجاع عدد الأجزاء في المكتبة

if __name__ == '__main__':  # إذا تم تشغيل الملف مباشرة
    # تأكد من وجود مجلد data/contrats_exemples وضع فيه ملف Contrat.pdf
    os.makedirs("data/contrats_exemples", exist_ok=True)  # إنشاء المجلد إذا لم يكن موجوداً
    initial_scan_and_build()  # مسح المجلدات وإضافة الملفات
    app.run(host='0.0.0.0', port=5000, debug=False)  # تشغيل الخادم على المنفذ 5000