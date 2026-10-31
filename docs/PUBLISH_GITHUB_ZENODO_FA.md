# راهنمای انتشار نهایی در GitHub و Zenodo

**قبل از انتشار عمومی** از موافقت هم‌نویسندگان، حقوق انتشار و مجوز (فایل `docs/LICENSE_DECISION_REQUIRED.md`) اطمینان حاصل کن.

## ۱. ساخت مخزن در GitHub

در GitHub با نام پیشنهادی `Decentralized-NMPC-Tug-Assisted-Berthing` مخزن جدید بساز. اگر هنوز انتشار مجاز نشده، مخزن را Private قرار بده. گزینه ساخت خودکار README یا .gitignore را خاموش کن؛ فایل‌های کامل در همین بسته موجودند.

## ۲. اجرا و اعتبارسنجی روی سیستم خودت

از PowerShell در ریشه پوشه:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe Scripts/verify_repository.py
```

## ۳. فعال‌سازی Git و اولین Push

```powershell
git init -b main
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
git add .
git status
git commit -m "Release reproducible research software and data"
git remote add origin https://github.com/YOUR-USERNAME/Decentralized-NMPC-Tug-Assisted-Berthing.git
git push -u origin main
```

نام کاربری و ایمیل را خودت وارد کن. از credential manager برای احراز هویت استفاده کن و رمز/توکن را داخل پروژه ذخیره نکن.

## ۴. بازتولید از یک Clone تمیز

پوشه جدید بساز و `git clone` بگیر، سپس آزمون‌های مرحله ۲ را دوباره اجرا کن. `Data/RawRuns` و فایل `Data/SHA256_MANIFEST.csv` باید کامل باشند.

## ۵. اصلاح متادیتا

آدرس رسمی GitHub را به `CITATION.cff` اضافه کن. اسامی نویسندگان، وابستگی‌ها و حق انتشار را تأیید کن. فایل `.zenodo.json` پیش‌نویس فراداده برای Zenodo است. یک مجوز مورد توافق را پیش از انتشار انتخاب کن.

## ۶. انتشار نسخه دقیق مقاله

پس از تأیید عمومی‌کردن، در صفحه GitHub به `Releases > Draft a new release` برو و برچسب `v1.0.0` و عنوان `Research software for the Ocean Engineering manuscript` را ثبت کن. توضیح Release در `docs/RELEASE_NOTES_TEMPLATE.md` آماده است.

## ۷. بایگانی DOI در Zenodo

در https://zenodo.org حساب GitHub را وصل کن. در بخش GitHub، مخزن را فعال کن **قبل از انتشار Release**؛ سپس Release را منتشر کن. بایگانی موفق و DOI اختصاصی را در Zenodo بررسی کن. اگر Release از قبل منتشر شده و به‌طور خودکار بایگانی نشده، رویه رسمی Zenodo برای بازبایگانی/انتشار رکورد را دنبال کن؛ DOI را حدس نزن.

## ۸. اصلاح Data availability در مقاله

فایل `docs/DATA_AVAILABILITY_TEMPLATE.tex` را با URL واقعی و DOI تأییدشده تکمیل کن و همان متن را در بخش مربوطه LaTeX مقاله جایگزین کن.

## ۹. بازبینی نهایی

- URL مخزن از مرورگر ناشناس باز شود.
- GitHub Release به commit مورد ارجاع مقاله اشاره کند.
- DOI Zenodo به همان Release و بسته داده مرتبط باشد.
- اعداد و شکل‌های مقاله با Data و Figures سازگار باشند.
- هیچ کلید دسترسی، اطلاعات خصوصی یا فایل بدون مجوز منتشر نشده باشد.
