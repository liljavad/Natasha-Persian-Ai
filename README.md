<div align="center">

# Project Natasha 🤖
### A Novel Framework for Persian Conversational AI
<p>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python" alt="Language: Python">
  <img src="https://img.shields.io/badge/Library-Hugging_Face-FFD000?style=for-the-badge&logo=huggingface" alt="Library: Hugging Face">
  <img src="https://img.shields.io/badge/Framework-Flask-000000?style=for-the-badge&logo=flask" alt="Framework: Flask">
  <img src="https://img.shields.io/badge/Status-In_Development-purple?style=for-the-badge" alt="Status: In Development">
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge" alt="License: Proprietary">
</p>

**[English](#english) | [فارسی (Persian)](#persian-farsi)**

</div>

---

<p align="center">
    <img src="https://imgur.com/Qbj70Wt.png" alt="Natasha UI Demo" style="width:100%; max-width:700px; border-radius:15px; border: 1px solid #333;"/>
</p>

---

## <a name="english"></a>🇬🇧 English

### Abstract

Project Natasha is a research and development initiative focused on creating a high-efficacy conversational AI specifically for the **Persian language**. Addressing the challenges of data scarcity and the morpho-syntactic complexity of Persian, this project introduces a novel pipeline centered around a **Conceptual Tokenizer**. This method creates a semantic abstraction layer by performing unsupervised clustering on word embeddings, allowing a foundational model to learn deeper contextual patterns from a limited dataset. A pre-trained T5-based transformer model was fine-tuned on this conceptual representation, resulting in a lightweight, efficient, and contextually aware chatbot.

### Core Methodology

The Natasha framework is built upon a unique multi-stage pipeline that differentiates it from standard fine-tuning approaches.

#### 🧠 The Concept Engine
The cornerstone of this project is a proprietary data processing pipeline we term the "Concept Engine." Instead of processing raw lexical units, the engine maps them to a condensed semantic space.

1.  **Embedding Generation:** A pre-trained multilingual model (`Sentence-Transformers`) is used to generate high-dimensional semantic vectors for each unique token.
2.  **Unsupervised Clustering:** The K-Means algorithm groups semantically similar tokens into a predefined number of clusters, where each cluster represents a distinct "concept."
3.  **Corpus Transformation:** The entire training dataset is then transformed by replacing each token with its corresponding Concept ID (e.g., `C56`).

This transformation results in a dataset where the model learns relationships between abstract ideas rather than literal words, leading to more robust pattern recognition and superior data efficiency.

#### ⚙️ Model Architecture and Fine-Tuning
We employed a **Transfer Learning** strategy using the `t5-small` model as our foundation. This powerful and efficient Text-to-Text Transformer was then fine-tuned on the conceptually transformed Persian conversational dataset.

### 🔬 A Glimpse into the "Concept Engine"
This is the core logic that translates words into abstract concepts. Instead of seeing individual words, the model learns from the underlying ideas, making it incredibly data-efficient.
<details>
<summary>Click to see a Python code snippet</summary>

```python
# Load the unique vocabulary from the preprocessed data
words = list(vocab_data.keys())
print(f"Embedding {len(words)} unique tokens...")

# Use a powerful multilingual model to generate semantic vectors
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
word_embeddings = model.encode(words, show_progress_bar=True)

print("Clustering vectors into semantic concepts using KMeans...")
# Group similar word vectors into N distinct concepts
kmeans = KMeans(n_clusters=NUM_CONCEPTS, random_state=42, n_init=10)
kmeans.fit(word_embeddings)

# Create the final map from a word to its abstract Concept ID
word_to_concept = {word: int(cluster_id) for word, cluster_id in zip(words, kmeans.labels_)}

print("✅ Conceptual map created successfully!")
```
</details>

### 🛠️ Technology Stack

| Category      | Technologies                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------|
| **AI/ML** | `Python`, `Hugging Face Transformers`, `PyTorch`, `Sentence-Transformers`, `Scikit-learn`         |
| **Backend** | `Flask`, `Gunicorn`                                                                           |
| **Frontend** | `HTML5`, `CSS3` (with animations & glassmorphism), `JavaScript` (Live streaming & dynamic UI) |
| **Persian NLP** | `Hazm`                                                                                        |

### 🗺️ Project Roadmap

Project Natasha is currently under active development with a focus on research, refinement, and performance optimization.

-   **[✅] Phase 1:** Development of the Concept Engine - *Completed*
-   **[✅] Phase 2:** Training of the initial prototype model - *Completed*
-   **[✅] Phase 3:** Development of the advanced, dynamic web interface - *Completed*
-   **[🚀] Phase 4:** Model optimization and preparation for demonstration - *In Progress*

#### Public Release Outlook (Q1 2026)
Our goal is to release the first interactive **Public Demo** of Project Natasha in the first quarter of 2026. This demo will allow users to experience the capabilities of our conceptually-driven model firsthand. Further announcements regarding potential API access for research or commercial purposes will follow the demo release.

### 💬 Community & Communication

This is a proprietary project, but community feedback and questions are highly valued. The best way to communicate with us is right here on GitHub.

-   **Have a question or a new idea?** 👉 [**Start a new Discussion**](https://github.com/liljavad/Natasha-Persian-Ai.git/discussions)
-   **Found a bug or have a feature request?** 👉 [**Open an Issue**](https://github.com/liljavad/Natasha-Persian-Ai.git/issues)
-   **Want to stay updated?** 👉 **Star (⭐)** and **Watch (👁️)** this repository!

We actively monitor these channels and look forward to hearing from the community.

---
---

## <a name="persian-farsi"></a>🇮🇷 فارسی (Persian)

### چکیده

پروژه ناتاشا یک طرح تحقیق و توسعه است که بر ساخت یک هوش مصنوعی محاوره‌ای با کارایی بالا، به طور خاص برای **زبان فارسی**، تمرکز دارد. این پروژه برای مقابله با چالش‌های کمبود داده و پیچیدگی‌های ساختاری زبان فارسی، یک پایپ‌لاین نوآورانه مبتنی بر **«توکنایزر مفهومی»** را معرفی می‌کند. این روش با انجام خوشه‌بندی بدون نظارت بر روی بازنمایی‌های برداری کلمات، یک لایه انتزاعی معنایی ایجاد می‌کند که به یک مدل پایه اجازه می‌دهد الگوهای عمیق‌تری را از یک دیتاست محدود بیاموزد. یک مدل ترنسفورمر از پیش‌آموزش‌دیده مبتنی بر T5 بر روی این داده‌های مفهومی بازآموزی (Fine-tune) شده که نتیجه آن، یک چت‌بات سبک، کارآمد و آگاه به زمینه گفتگو است.

### متدولوژی اصلی

چارچوب ناتاشا بر یک پایپ‌لاین چندمرحله‌ای منحصربه‌فرد بنا شده است که آن را از رویکردهای استاندارد بازآموزی متمایز می‌کند.

#### 🧠 موتور مفهومی (Concept Engine)
سنگ بنای این پروژه، پایپ‌لاین پردازش داده اختصاصی ماست که آن را «موتور مفهومی» می‌نامیم. این موتور به جای پردازش کلمات خام، آن‌ها را به یک فضای معنایی فشرده نگاشت می‌کند.

1.  **تولید بازنمایی برداری:** یک مدل چندزبانه (`Sentence-Transformers`) برای تولید وکتورهای معنایی برای هر کلمه یکتا استفاده می‌شود.
2.  **خوشه‌بندی بدون نظارت:** الگوریتم K-Means وکتورهای کلماتِ با معنای مشابه را در خوشه‌هایی گروه‌بندی می‌کند که هر خوشه نماینده یک «مفهوم» مجزاست.
3.  **تبدیل دیتاست:** کل دیتاست آموزشی با جایگزینی هر کلمه با شناسه مفهومی متناظر آن (مثلاً `C56`) بازنویسی می‌شود.

این تحول باعث می‌شود مدل به جای یادگیری روابط بین کلمات، روابط بین ایده‌های انتزاعی را بیاموزد که به شناسایی الگوی قوی‌تر و بهره‌وری داده بالاتر منجر می‌شود.

#### ⚙️ معماری و بازآموزی مدل
ما از استراتژی **یادگیری انتقالی (Transfer Learning)** با استفاده از مدل `t5-small` به عنوان پایه بهره بردیم. این ترنسفورمر قدرتمند سپس بر روی دیتاست مفهومی‌سازی‌شده فارسی، بازآموزی شد.


### 🔬 نگاهی به کد «موتور مفهومی»
این بخش، منطق اصلی ترجمه کلمات به مفاهیم انتزاعی را نشان می‌دهد. مدل به جای دیدن کلمات مجزا، از ایده‌های بنیادین آن‌ها یاد می‌گیرد که این فرآیند را به طرز شگفت‌انگیزی بهینه می‌کند.
<details>
<summary>برای مشاهده نمونه کد پایتون کلیک کنید</summary>

```python
# بارگذاری واژگان یکتا از داده‌های پیش‌پردازش شده
words = list(vocab_data.keys())
print(f"در حال ساخت امبدینگ برای {len(words)} توکن یکتا...")

# استفاده از یک مدل قدرتمند چندزبانه برای تولید وکتورهای معنایی
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
word_embeddings = model.encode(words, show_progress_bar=True)

print("در حال خوشه‌بندی وکتورها به مفاهیم معنایی با KMeans...")
# گروه‌بندی وکتورهای کلمات مشابه در N مفهوم مجزا
kmeans = KMeans(n_clusters=NUM_CONCEPTS, random_state=42, n_init=10)
kmeans.fit(word_embeddings)

# ساخت نقشه نهایی از هر کلمه به شناسه مفهومی انتزاعی آن
word_to_concept = {word: int(cluster_id) for word, cluster_id in zip(words, kmeans.labels_)}

print("✅ نقشه مفهومی با موفقیت ایجاد شد!")
```
</details>

### 🛠️ پشته فناوری (Technology Stack)

| دسته          | فناوری‌ها                                                                                    |
|---------------|-----------------------------------------------------------------------------------------------|
| **هوش مصنوعی**| `Python`, `Hugging Face Transformers`, `PyTorch`, `Sentence-Transformers`, `Scikit-learn`         |
| **بک‌اند** | `Flask`, `Gunicorn`                                                                           |
| **فرانت‌اند** | `HTML5`, `CSS3` (همراه با انیمیشن و گلس‌مورفیسم), `JavaScript` (تایپ زنده و UI پویا) |
| **پردازش فارسی**| `Hazm`                                                                                        |

### 🗺️ نقشه راه پروژه

پروژه ناتاشا در حال حاضر در فاز توسعه فعال با تمرکز بر تحقیق، بهبود متدولوژی و بهینه‌سازی عملکرد قرار دارد.

-   **[✅] فاز ۱:** توسعه موتور مفهومی - *پایان یافته*
-   **[✅] فاز ۲:** آموزش مدل اولیه - *پایان یافته*
-   **[✅] فاز ۳:** طراحی رابط کاربری پیشرفته و پویا - *پایان یافته*
-   **[🚀] فاز ۴:** بهینه‌سازی مدل و آماده‌سازی برای دمو - *در حال انجام*

#### چشم‌انداز انتشار عمومی (سه‌ماهه اول ۲۰۲۶)
هدف ما انتشار اولین **دموی عمومی** و تعاملی پروژه ناتاشا در سه‌ماهه اول سال ۲۰۲۶ است. این دمو به کاربران اجازه می‌دهد تا قابلیت‌های مدل مفهوم‌محور ما را به صورت زنده تجربه کنند. اطلاعات بیشتر در مورد دسترسی احتمالی به API برای اهداف تحقیقاتی یا تجاری، پس از انتشار دمو اعلام خواهد شد.

### 💬 ارتباط و جامعه کاربری

این یک پروژه انحصاری است، اما ما برای بازخورد و سوالات جامعه کاربری ارزش زیادی قائل هستیم. بهترین راه برای ارتباط با ما، همین‌جا در گیت‌هاب است.

-   **سوال یا ایده جدیدی دارید؟** 👈 [**یک گفتگوی جدید در بخش Discussions آغاز کنید**](https://github.com/liljavad/Natasha-Persian-Ai.git/discussions)
-   **باگ پیدا کردید یا درخواست قابلیتی دارید؟** 👈 [**یک ایشو (Issue) جدید باز کنید**](https://github.com/liljavad/Natasha-Persian-Ai/issues)
-   **می‌خواهید از آخرین اخبار مطلع شوید؟** 👈 این ریپازیتوری را **استار (⭐)** و **واچ (👁️)** کنید!

ما این کانال‌ها را به طور فعال بررسی می‌کنیم و مشتاقانه منتظر شنیدن نظرات شما هستیم.
