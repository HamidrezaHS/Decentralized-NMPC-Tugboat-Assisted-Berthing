# رفع خطای HASH MISMATCH در GitHub Actions

## علت

فایل‌های عددی CSV با پایان خط ویندوز (`CRLF`) بایگانی شده‌اند، اما Git ممکن است هنگام Commit یا Checkout پایان خطوط را به `LF` تبدیل کند. مانیفست `Data/SHA256_MANIFEST.csv` شامل SHA-256 **بایت‌های اولیه** است. تغییر فقط پایان خطوط باعث می‌شود بررسی خام شکست بخورد، بدون آنکه مقادیر عددی تغییر کرده باشند.

## اصلاح این بسته

- `Scripts/verify_repository.py` همچنان SHA-256 و اندازه اولیه هر فایل را بررسی می‌کند.
- در مورد CSVها *فقط* اجازه می‌دهد بازنمایی یکسان با `LF` یا `CRLF` نیز با هر دو مقدار مانیفست اولیه تطبیق پیدا کند.
- هر تغییر واقعی در اعداد یا محتوای داده همچنان باعث شکست می‌شود.
- فایل `.gitattributes` برای جلوگیری از تغییر پایان خط CSVهای آرشیو در Commitهای آینده اضافه شده است.
- آزمون‌های مستقل در `tests/test_repository_hash_endings.py` رفتار محافظتی این اصلاح را بررسی می‌کنند.

**نیازی به تغییر مانیفست یا فایل‌های CSV موجود در GitHub نیست.** به‌ویژه برای رفع ظاهری هش، فایل `SHA256_MANIFEST.csv` را از نو بر اساس داده‌های تغییریافته تولید نکنید؛ این کار زنجیره اصالت آرشیو را از بین می‌برد.

## ساده‌ترین روش با GitHub Desktop

1. فایل `GitHub_Hash_Check_Fix_Only.zip` را استخراج کنید.
2. پوشه مخزن محلی را در GitHub Desktop با **Repository → Show in Explorer** باز کنید.
3. فایل‌های داخل بسته اصلاحی را با **ساختار پوشه‌ای یکسان** در ریشه مخزن کپی کنید. وقتی سؤال شد، نسخه جدید `Scripts/verify_repository.py` را جایگزین کنید. فایل `.gitattributes` را نیز در ریشه مخزن کپی کنید. آزمون جدید را در `tests/` بگذارید.
4. در GitHub Desktop به بخش **Changes** بروید و دقت کنید فقط فایل‌های اصلاحی تغییر کرده باشند. اگر فایل‌های متعدد CSV نیز در Changes ظاهر شدند، برای این اصلاح نیازی به Commit آن‌ها نیست؛ تیک آن‌ها را بردارید.
5. در قسمت Summary بنویسید: `Fix cross-platform CSV archive integrity verification`.
6. **Commit to main** و سپس **Push origin** را بزنید.
7. در سایت GitHub وارد مخزن و سپس تب **Actions** شوید. اجرا باید با تغییر جدید به‌طور خودکار شروع شود. در صورت موفقیت، بخش `Verify archived data` سبز می‌شود.

## آزمون دستی در PowerShell

در ریشه مخزن اجرا کنید:

```powershell
python Scripts/verify_repository.py
python -m pytest -q tests
```

پیام موفقیت نهایی معمولاً چنین است:

```text
Integrity: 129 archived data files checked; 23 cases indexed.
INFO: 53 CSV files match the archived SHA-256 after strictly verified CRLF/LF line-ending conversion.
PASS: all archived files match their recorded content, and all 23 simulation cases are present.
```

خط `INFO` ممکن است ظاهر نشود، اگر فایل‌ها هنوز با بایت‌های اولیه یکسان باشند. هر دو حالت معتبرند.
