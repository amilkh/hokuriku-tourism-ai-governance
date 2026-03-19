<div align="center">

# إطار حوكمة الذكاء الاصطناعي للسياحة في هوكوريكو

### This is my own translation and not professional, also I am not native speaker, or any speaker so it might have some mistakes. I was translating using my friend and google translate
### التنبؤ بالطلب السياحي وتحليل الضعف المكاني في الحيوية بالاعتماد على الذكاء الاصطناعي

**Amil Khanzada** — *أستاذ معيّن بصفة خاصة، مختبر الإنعاش الإقليمي، جامعة فوكوي*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Data Validated](https://img.shields.io/badge/rows%20audited-1.4M-brightgreen.svg)](src/validator.py)

> **التقارير التنفيذية:**
> [English](EXECUTIVE_REPORT.en.md) ·
> [日本語](EXECUTIVE_REPORT.ja.md)
>
> **اقرأ أيضاً:** [English](README.md) · [日本語](README.ja.md)

</div>

---

## الملخص

ينفّذ هذا المستودع محرك البيانات البشرية الموزعة **DHDE**، وهو إطار بحثي يدمج مصادر بيانات سياحية غير متجانسة تشمل بيانات تدفق الزوار من كاميرات الذكاء الاصطناعي، ومشاهدات الأرصاد الجوية من وكالة الأرصاد اليابانية JMA، وإشارات النية من Google Business Profile، إضافة إلى 95,653 استجابة من استبيانات هوكوريكو، ضمن خط تحليل تنبؤي وتشخيصي موحّد.

يقيس النظام العجز الهيكلي في السياحة بمحافظة فوكوي: **فجوة الفرصة السنوية البالغة 11.96 مليار ين**، وهي الإيرادات المفقودة بسبب كبح الطلب الناتج عن سوء الأحوال الجوية في أشهر الشتاء، عندما تحتل فوكوي المرتبة **47 من أصل 47 محافظة** على المستوى الوطني.

**الكلمات المفتاحية:** التنبؤ بالطلب السياحي · هندسة كانسي · مؤشر عدم الارتياح · التشبع المكاني · ضعف الحيوية · حوكمة إقليم هوكوريكو

---

## 1. الإطار النظري: محرك البيانات البشرية الموزعة (DHDE)

يدمج DHDE أربع طرائق استشعار في خط تحليلي واحد:

```
┌─────────────────────────────────────────────────────────────────┐
│                  DISTRIBUTED HUMAN DATA ENGINE                  │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ AI Camera  │  │ JMA Weather│  │ Google BizP │  │ Hokuriku │ │
│  │ People-Flow│  │ 8-Variable │  │ Route Intent│  │  Survey  │ │
│  │  (Edge-AI) │  │  (Hourly)  │  │   (Daily)   │  │ (95,653) │ │
│  └─────┬──────┘  └─────┬──────┘  └──────┬──────┘  └────┬─────┘ │
│        │               │                │              │       │
│        └───────────┬───┴────────────────┴───────┬──────┘       │
│                    │      هندسة الخصائص         │              │
│                    │ التقويم · شدة الطقس · اللواحق │           │
│                    │     النوافذ · التفاعلات     │              │
│                    └──────────────┬──────────────┘              │
│                                   │                             │
│              ┌────────────────────┴───────────────────┐         │
│              │ الانحدار الخطي OLS + Random Forest     │         │
│              │ DW, NW-HAC, FD, LDV, VIF              │         │
│              └────────────────────┬───────────────────┘         │
│                                   │                             │
│         ┌─────────────────────────┼────────────────────────┐    │
│         │                         │                        │    │
│  ┌──────▼──────┐  ┌──────────────▼──────────┐  ┌──────────▼──┐ │
│  │ Opportunity │  │ Kansei Assessment       │  │  Spatial    │ │
│  │ Gap / Lost  │  │ DI · WC · Overtourism   │  │ Saturation  │ │
│  │ Population  │  │ Text Mining (NLP)       │  │ Multi-Node  │ │
│  └─────────────┘  └─────────────────────────┘  └─────────────┘ │
│                                                                 │
│                   ──► analysis_metrics.txt                      │
│                   ──► LaTeX tables for paper                    │
└─────────────────────────────────────────────────────────────────┘
```

**العقد في الشبكة المكانية:**

| العقدة | الموقع | مصدر الكاميرا | محطة الطقس |
|--------|--------|---------------|------------|
| A | توجينبو | tojinbo-shotaro | Mikuni (JMA) |
| B | شرق محطة فوكوي | fukui-station-east | Fukui (JMA) |
| C | كاتسوياما | katsuyama | Katsuyama (JMA) |
| D | خط رينبو | rainbow-line-parking-lot-1-gate | Fukui (proxy) |

---

## 2. النتائج الرئيسية

| المؤشر | القيمة | التفسير |
|--------|--------|---------|
| **OLS R²** | 0.810 (Adj R² = 0.802) | القوة التفسيرية الأساسية |
| **RF 5-fold CV R²** | 0.557 ± 0.131 | دقة التنبؤ خارج العينة |
| **First-Difference R²** | 0.708 | بعد تصحيح الارتباط الذاتي |
| **LDV R² / DW** | 0.848 / 1.899 | نموذج ديناميكي مع بقايا نظيفة |
| **المتغير الأهم** | Google `directions` | نية البحث عن المسار، r = +0.781 |
| **إزاحة إيشيكاوا → توجينبو** | r = +0.549 | خط أنابيب طلب عابر للمحافظات |
| **الزوار مقابل الرضا** | rs = +0.150 (p = 0.002) | **لا يوجد فرط سياحة** |
| **الزوار المفقودون** | 85,522 | فجوة الفرصة لعقدة واحدة سنوياً |
| **حساسية الطقس الشتوية** | 6.26× الصيف | عدم تماثل موسمي |
| **نسبة ضعف الحيوية** | 11.5× | انتشار المراجعات منخفضة الرضا |
| **الترتيب الوطني شتاءً** | 47 / 47 | العجز الهيكلي لفوكوي |

---

## 3. فجوة الفرصة بقيمة 11.96 مليار ين

تقيس **فجوة الفرصة** الفرق بين عدد الزوار المتوقعين بناءً على إشارات نية Google وعدد القادمين الفعليين في الأيام المتدهورة مناخياً:

$$
\text{Lost Visitors}_d = \hat{y}_d^{\text{OLS}} - y_d^{\text{actual}} \quad \text{when} \quad y_d < \hat{y}_d
$$

$$
\text{Total Economic Loss} = \sum_{d \in \mathcal{G}} \text{Lost Visitors}_d \times \bar{S}
$$

حيث إن $\bar{S} = ¥13{,}811$ هو متوسط الإنفاق لكل زائر، و$\mathcal{G}$ هي مجموعة أيام الفجوة.

| المكوّن | القيمة |
|---------|--------|
| أيام الفجوة | 42 |
| إجمالي الزوار المفقودين | 85,522 |
| متوسط الإنفاق لكل زائر | ¥13,811 |
| **إجمالي الخسارة السنوية في الإيراد** | **¥11.96 billion** |

---

## 4. تقييم كانسي البيئي

### 4.1 مؤشر عدم الارتياح

$$
DI = 0.81 \cdot T + 0.01 \cdot H \cdot (0.99 \cdot T - 14.3) + 46.3
$$

حيث إن $T$ هي درجة الحرارة المئوية و$H$ هي الرطوبة النسبية.

### 4.2 برودة الرياح

$$
WC = 13.12 + 0.6215T - 11.37V^{0.16} + 0.3965TV^{0.16}
$$

حيث إن $V$ هي سرعة الرياح بالكيلومتر في الساعة. تنطبق الصيغة عندما يكون $T \leq 10^\circ C$ و$V > 4.8$ كم/س.

### 4.3 عتبة فرط السياحة

ارتباط سبيرمان بين عدد الزوار اليومي ومتوسط الرضا:

rs(visitors, satisfaction) = +0.150 (p = 0.002)

يشير هذا الارتباط الإيجابي إلى أن مشكلة فوكوي ليست فرط السياحة، بل **ضعف الحيوية**: المزيد من الزوار يقترن برضا أعلى.

---

## 5. خريطة التشبع المكاني

يحقق تحليل العقد المتعددة تشبعاً جغرافياً لمحافظة فوكوي:

```
              ┌──── العقدة C: كاتسوياما (جبلية / شرق) ────┐
              │                                           │
   العقدة A: توجينبو ─── العقدة B: محطة فوكوي ─── العقدة D: خط رينبو
   (ساحلية / شمال)        (حضرية / مركز)           (منظرية / جنوب)
```

يتم نمذجة كل عقدة محلياً باستخدام طقس JMA المحلي، ما يتيح:
- **شبكة الدرع المناخي**: عندما تكون ميكوني الساحلية عاصفة، قد تكون كاتسوياما الداخلية صافية
- **إعادة توزيع الطلب** عبر توجيه جوي لحظي

---

## 6. متانة النموذج

| التشخيص | الإحصائية | التفسير |
|---------|-----------|---------|
| Durbin-Watson (OLS) | 1.005 | تم تصحيحه عبر Newey-West HAC |
| Durbin-Watson (1st-diff) | 2.525 | **بقايا نظيفة** |
| Newey-West HAC | 8 significant | متين تجاه عدم تجانس التباين |
| First-Difference R² | 0.708 | يتحكم في الاتجاه العام |
| LDV R² | 0.848 | توصيف ديناميكي |
| VIF (max) | < 10 | لا تعدد ترابط خطير |
| قيمة بيانات الطقس | +0.056 R² | مساهمة JMA مقاسة |

---

## 7. بنية المستودع

```
hokuriku-tourism-ai-governance/
├── pyproject.toml                # تعريف الحزمة وفق PEP 517/621
├── requirements.txt              # تبعيات التشغيل الدنيا
├── config/
│   └── settings.yaml             # إعدادات المسارات والمعلمات
├── src/
│   ├── __init__.py
│   ├── config.py                 # تحميل الإعدادات وحل المسارات
│   ├── data_loader.py            # تحميل الكاميرات والطقس وGoogle والاستبيانات
│   ├── feature_engineering.py    # التقويم، شدة الطقس، اللواحق، التفاعلات
│   ├── models.py                 # OLS + Random Forest + اختبارات المتانة
│   ├── kansei.py                 # مؤشر عدم الارتياح، برودة الرياح، تنقيب النصوص
│   ├── economics.py              # فجوة الفرصة والخسارة السكانية والترتيب
│   ├── spatial.py                # الارتباطات المتقاطعة والتحليل متعدد العقد
│   ├── validator.py              # تدقيق نزاهة البيانات
│   ├── visualizer.py             # توليد جميع الأشكال
│   ├── latex_export.py           # توليد جداول LaTeX
│   ├── report.py                 # Reporter مركزي للتسجيل والقياسات
│   └── run_analysis.py           # نقطة دخول خط الأنابيب
├── tests/
│   ├── test_models.py
│   ├── test_kansei.py
│   ├── test_validator.py
│   ├── test_features.py
│   └── test_math.py
├── jma/                          # بيانات الطقس الملتزم بها داخل المستودع
├── EXECUTIVE_REPORT.en.md
├── EXECUTIVE_REPORT.ja.md
├── output/                       # نواتج مولدة مرجعية
├── README.md
├── README.ja.md
└── README.ar.md
```

---

## 8. مصادر البيانات

| المصدر | النوع | التغطية | الصفوف |
|--------|-------|---------|--------|
| **AI Camera** | عدّ أشخاص على فواصل 5 دقائق | 2024-12 → 2026-02 | ~170K |
| **JMA** | طقس ساعي: مطر، حرارة، شمس، رياح، رطوبة، ثلوج | 2024-01 → 2026-02 | ~140K |
| **Google Business Profile** | مقاييس يومية للمسارات والمشاهدات والمراجعات | 2024-01 → 2026-02 | ~35K |
| **Hokuriku Tourism Survey** | رضا، NPS، ونص حر | 2023 → 2026 | **95,653** |
| **Fukui Kanko Survey (raw)** | إنفاق، ديموغرافيا، أنماط سفر | 2022 → 2025 | ~1M |

**إجمالي الصفوف المدققة بواسطة validator.py:** نحو 1.4 مليون صف.

---

## 9. خطوات إعادة الإنتاج

### الإعداد

```bash
# أنشئ مساحة عمل مع المستودعات الشقيقة للبيانات
mkdir hokuriku-workspace && cd hokuriku-workspace
git clone https://github.com/code4fukui/fukui-kanko-people-flow-data.git
git clone https://github.com/code4fukui/fukui-kanko-trend-report.git
git clone https://github.com/code4fukui/opendata.git
git clone https://github.com/code4fukui/fukui-kanko-survey.git

# استنسخ هذا المستودع وثبّته
git clone https://github.com/amilkh/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance
pip install ".[dev]"
```

### الأوامر

| الأمر | ما الذي يفعله |
|-------|---------------|
| `python -m src.run_analysis` | تشغيل خط الأنابيب الكامل وإنتاج الأشكال والقياسات وجداول LaTeX |
| `pandoc EXECUTIVE_REPORT.en.md --pdf-engine=xelatex -o output/pdf/executive_report_en.pdf` | بناء التقرير التنفيذي الإنجليزي PDF |
| `pandoc EXECUTIVE_REPORT.ja.md --pdf-engine=xelatex -o output/pdf/executive_report_ja.pdf` | بناء التقرير التنفيذي الياباني PDF |
| `pytest` | تشغيل مجموعة الاختبارات |
| `pytest --cov=src --cov-report=html` | تشغيل الاختبارات مع تقرير التغطية |
| `ruff check src/ tests/` | فحص الأسلوب واللينت |

> **متطلبات PDF:** `sudo apt-get install -y pandoc texlive-xetex texlive-lang-japanese fonts-noto-cjk`
>
> **ملاحظة:** اضبط `HTAG_CONFIG=/path/to/settings.yaml` لاستخدام إعداد مخصص، والافتراضي هو `config/settings.yaml`.

تُكتب جميع النواتج إلى `output/`: الأشكال، نسخ EN وJA، جداول LaTeX، التقارير التنفيذية، وملفات PDF المجمعة.

---

## 10. البنية المعيارية

يتبع خط الأنابيب **فصلاً صارماً للمسؤوليات**:

```python
# Entrypoint: src/run_analysis.py
cfg = load_config()                           # config.py
rpt = Reporter(cfg)                           # report.py
validation = validate_pipeline(cfg, rpt)      # validator.py
data = load_all_data(cfg, rpt)                # data_loader.py
daily, features = build_features(daily, ..)   # feature_engineering.py
ols = fit_ols(model_df, features, rpt)        # models.py
rf  = fit_random_forest(model_df, ..)         # models.py
robust = robustness_suite(model_df, ..)       # models.py
gap = compute_opportunity_gap(daily, ..)      # economics.py
kansei = discomfort_index_analysis(..)        # kansei.py
spatial = multi_node_analysis(cfg, ..)        # spatial.py
export_all_tables(results, ..)                # latex_export.py
```

تستقبل كل وحدة كائناً من Reporter لتسجيل حتمي. لا ينبغي لأي وحدة استخدام `print()` مباشرة؛ بل يمر كل المخرج عبر المراسل المركزي.

---

## 11. الاختبار والتحقق

### مجموعة الاختبارات

```
tests/
├── test_models.py     # OLS R², RF importance, DW, edge cases
├── test_kansei.py     # DI hand-calculations, wind chill, golden values
├── test_validator.py  # Schema, outliers, date gaps, drift detection
├── test_features.py   # Calendar, severity, lags, encodings
└── test_math.py       # Core statistical function correctness
```

### التحقق من البيانات (`src/validator.py`)

يقوم النظام تلقائياً بتدقيق كل مصدر بيانات من أجل:
- **اختلالات المخطط**: أعمدة أضيفت أو أزيلت بين نسخ البيانات
- **انجراف البيانات**: اختبارات Kolmogorov-Smirnov على نوافذ انزلاقية مدتها 3 أشهر
- **القيم الشاذة**: كشف IQR وZ-score لكل عمود
- **فجوات التواريخ**: الأيام المفقودة في السلاسل الزمنية
- **انتهاكات المجال**: أمطار سالبة أو درجات حرارة متطرفة

تُضمّن النتائج في `output/analysis_metrics.txt`.

---

## الترخيص

حقوق النشر © 2026 Amil Khanzada. جميع الحقوق محفوظة.

لا يُمنح أي ترخيص حالياً. تتطلب إعادة الاستخدام أو إعادة التوزيع أو النشر إذناً كتابياً صريحاً من المؤلف.