# راهنمای فارسی مخزن پژوهشی

این مخزن شامل نسخه اجرایی کنترل پیش‌بین مدل غیرمتمرکز برای پهلوگیری KVLCC2 با دو یدک‌کش فشاری و دو یدک‌کش کششی، همراه با داده‌های اصلی مقاله است.

## چه چیزهایی قابل بازتولید است؟

- ۱۶ ترکیب جهت باد و جریان در هدینگ اولیه ۲۰ درجه؛
- ۶ سناریوی بازگشتی و یک سناریوی ترکیبی با هدینگ ۴۰ درجه؛
- شکل‌های مقاله، مسیر کشتی، فرمان‌های یدک‌کش‌ها و خلاصه آزمون‌ها.

موفقیت ثبت‌شده در این اجراها به معنای ورود به محدوده پذیرش اولیه پهلوگیری است، نه اثبات پایداری طولانی‌مدت یا ایمنی در تمامی شرایط.

## نصب و آزمون سریع در PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe Scripts/verify_repository.py
```

## بازسازی نمودارهای مقاله

```powershell
.\.venv\Scripts\python.exe Scripts/make_publication_figures.py
```

## ممیزی نتایج جهت‌های باد و جریان

```powershell
.\.venv\Scripts\python.exe ENV_DIRECTION_VERIFY.py --results Data/RawRuns/environment_4x4_v31
```

## انتشار در GitHub و Zenodo

مراحل انتشار و تنظیم DOI در `docs/PUBLISH_GITHUB_ZENODO_FA.md` قرار دارد. تا زمانی که حق انتشار کد و مجوز نرم‌افزار تأیید نشده، مخزن را خصوصی نگه دارید. فایل‌های `Data/SHA256_MANIFEST.csv` و `Data/RECORDED_CASES.csv` را قبل از انتشار کنترل کنید.

## رسم ماتریس نتایج بدون دست‌زدن به داده خام

```powershell
.\.venv\Scripts\python.exe Scripts/plot_directional_archive.py
```
