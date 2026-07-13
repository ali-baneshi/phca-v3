# PHCA v3.0 — بازبینیِ دورِ دوم (بعد از ۹ فیکسِ اعمال‌شده)

> مبتنی بر commit `0abef3b` (`PHCA-Maturation-Hardening-finish-round-25`).
> مرجع: `phca-v3-findings.md` (دورِ اول). این سند فقط چیزهای *جدید* را پوشش می‌دهد.

---

## بخش ۱: تأییدِ فیکس‌ها (خط‌به‌خط، از روی کد)

### ✅ ۱. Attention → Learning (رفعِ F-01) — **درست و کامل**
`world_model/mlp.py`، `learn()`: حالا `per_dim_weights=self._attention_weights` واقعاً به `_backward` و `_backward_batch` پاس داده می‌شود و `d_out` را ضرب می‌کند (خطوط ۴۱۶، ۴۳۵، ۵۷۲-۵۷۳، ۶۷۱-۶۷۲). گرادیان واقعاً بر اساس وزنِ توجه تعدیل می‌شود.
- **نکته‌ی ریز (نه باگ، صرفاً یک ملاحظه برای آینده):** در فازِ replay، `self._attention_weights` وزنِ *سیکل جاری* است ولی به همه‌ی نمونه‌های minibatch (که از سیکل‌های گذشته‌اند) یکسان اعمال می‌شود. یعنی وزنِ توجه دیگر «متعلق به لحظه‌ی خودش» نیست، بلکه یک نویز/بایاسِ سراسری روی کل replay است. اثرش احتمالاً کوچک است ولی اگر روزی خواستید attention را دقیق‌تر کنید، این یک نامزدِ بهبود است (ذخیره‌ی attention_weights به‌ازای هر نمونه در replay buffer).

### ✅ ۲. حذفِ hard-override در MDIM (رفعِ F-05) — **درست و طراحیِ خوبی دارد**
`motivation/mdim.py`، `generate_goal()`: دیگر `if task_lock: winner=1` وجود ندارد. به‌جایش `goal_switch_boost` فقط یک nudge نرم به deficit درایوِ D1 می‌دهد (`max(deficit, 2.0)`) و softmax همیشه واقعاً اجرا می‌شود. این دقیقاً همان اصلاحی است که باید انجام می‌شد — منطقی‌تر از حذفِ کاملِ signal هم هست (چون حالا رویدادِ «هدف جابه‌جا شد» هنوز به‌طور معناداری در سیستم اثر دارد، ولی به‌شکلِ رقابتی نه دیکتاتوری).

### ✅ ۳. رفعِ bypass هندسی در انتخابِ عمل (رفعِ F-06 بخشِ اول) — **درست، و ظریف**
`core/cycle.py`، `_select_action()`: حلقه‌ی امتیازدهی روی G′ حالا **همیشه** اجرا می‌شود (دیگر `return` زودهنگام برای task-lock نیست). `geo_action` به‌عنوان یک prior اضافه می‌شود با وزنِ `w_geo = max(0.0, 0.5 - conf)` — یعنی فقط وقتی اطمینان زیرِ ۰.۵ است اثر دارد و با رشدِ اطمینان محو می‌شود. این طراحیِ درستی است: هندسه دیگر جایگزینِ پیش‌بینی نیست، فقط یک prior برای حالتِ عدم‌قطعیت است.

### ✅ ۴. `--disable-task-lock` + پیش‌فرضِ L3 — **درست**
`scripts/benchmark.py` خط ۹۶-۹۷: `if level == 3 and level_iv is None: level_iv = InterventionConfig(disable_task_lock=True)`. این دقیقاً مشکلِ خاصِ L3 (که قبلاً هیچ‌وقت واقعاً «بدون هدف» نبود) را حل می‌کند.

### ✅ ۵. متریکِ `eval_prediction_error` در L4 — **بهبودِ واقعی**
`scripts/benchmark_level4.py`: حالا در کنارِ `goal_rate` (رفتاری)، یک متریکِ سطحِ مدل (خطای پیش‌بینیِ روی تسک‌های قدیمی حینِ eval) هم گزارش می‌شود. این دقیقاً پاسخ به نقدِ F-06 است — گزارش، دیگر تک‌بعدی نیست.

### ✅ ۸. Dedup در ساختِ گراف/G′ — **بهبودِ بهداشتی درست**
`_build_gprime` و `_build_discrete_graph_nodes` استخراج شده‌اند و در هر دو مسیرِ ساخت استفاده می‌شوند (خطوط ۲۳۲۹-۲۴۷۰). کاهشِ ریسکِ drift.

---

## بخش ۲: یافته‌ی جدید و مهم — رگرسیون در RBTA

### 🔴 NEW-01 — فیکسِ «شدت-محورِ» RBTA به‌طور کامل enforcement کفِ آنتروپی (A3) را غیرفعال کرده
**شدت: بحرانی — این خودِ همان invariant (A3) است که پروژه در سراسر مستندات به آن افتخار می‌کند.**

ریشه‌ی مشکل در `config.py`، `ConstraintViolation.__post_init__`:
```python
def __post_init__(self):
    if self.allowed > 0 and self.measured > self.allowed:
        ratio = (self.measured - self.allowed) / max(self.allowed, 1e-9)
        self.severity = float(np.clip(ratio / 5.0, 0.0, 1.0))
    else:
        self.severity = 0.0
```
این فرمول فرض می‌کند «نقض» همیشه یعنی `measured > allowed` (خیلی زیاد — درست برای TIME/MEM/ENERGY/SENSOR). اما برای ENTROPY_FLOOR، جهتِ نقض **برعکس** است: نقض یعنی `measured < allowed` (آنتروپی خیلی *کم*، یعنی مدل بیش‌ازحد مطمئن/overconfident شده — دقیقاً همان چیزی که A3 می‌خواهد جلویش را بگیرد).

در `rbta_enforcer.py` خط ۱۶۲-۱۶۸:
```python
if entropy < bounds.entropy_floor:
    violations.append(ConstraintViolation(
        module_id=module_id, bound_type="ENTROPY",
        measured=entropy, allowed=bounds.entropy_floor,
    ))
```
این‌جا `measured` (=entropy) همیشه کوچک‌تر از `allowed` (=floor) است — یعنی شرطِ `measured > allowed` در `__post_init__` **همیشه False** می‌شود → `severity` همیشه `0.0` می‌شود.

**پیامد در `_classify_action`:**
```python
weighted = sum(v.severity for v in violations)   # اگر تنها نقض، ENTROPY باشد → 0.0
...
elif weighted > 0.0:
    return EnforcerAction.INTERRUPT
else:
    return EnforcerAction.CONTINUE   # ← این‌جا فرود می‌آید
```
یعنی: **هر نقضِ کفِ آنتروپی که به‌تنهایی رخ دهد (بدون هم‌زمانی با یک نقضِ TIME/MEM/ENERGY)، اکنون به‌طور کامل توسط RBTA نادیده گرفته می‌شود** — نه فقط با شدتِ کمتر، بلکه **کاملاً بی‌اثر**، انگار اصلاً رخ نداده. این از نسخه‌ی قبل هم بدتر است: در نسخه‌ی قبلِ کد (شمارشِ خام)، یک نقضِ آنتروپیِ منفرد حداقل با «۱ نقض» به `INTERRUPT` منجر می‌شد؛ الان با severity=0 حتی همان هم رخ نمی‌دهد.

**چرا تست‌ها این را نگرفتند؟** رفتم `test_rbta.py` را چک کردم — دقیقاً تستِ مربوطه هست:
```python
def test_entropy_floor_violation(self, enforcer: RBTAEnforcer):
    violations, _ = enforcer.check_cycle(...belief_entropies={"G'": 0.001}...)
    assert len(violations) == 1
    assert violations[0].bound_type == "ENTROPY"
    ...
```
مقدارِ برگشتیِ دوم (`action`) با `_` دور ریخته شده و **هرگز assert نمی‌شود**! یعنی تست فقط تأیید می‌کند «یک شیءِ violation ساخته شد»، نه اینکه «سیستم واقعاً به آن واکنش نشان داد». این دقیقاً همان کورسوراخی است که باگ از آن رد شده.

**تأثیرِ عملی روی ادعای پروژه:** README و اسنادِ نظری A3 را این‌طور معرفی می‌کنند: «سیستم هیچ‌وقت اجازه نمی‌دهد باورش بیش‌ازحد قطعی شود (کفِ آنتروپی همیشه enforced است)». با این باگ، این جمله دیگر درست نیست برای هیچ‌کدام از ۱۰ ماژولی که تازه (در همین دورِ فیکس) به `belief_entropies` اضافه شدند — enforcement آن‌ها *measurement*‌اش درست کار می‌کند اما *reaction*‌اش کاملاً خاموش است.

#### راه‌حلِ پیشنهادی (دو گزینه)

**گزینه‌ی الف (ترجیح داده می‌شود — تمیزتر):** جهتِ نقض را در خودِ `ConstraintViolation` صریح کنید، چون `severity` نباید حدس بزند نقض کدام‌جهته:
```python
@dataclass
class ConstraintViolation:
    module_id: str
    bound_type: str
    measured: float
    allowed: float
    direction: str = "over"  # "over" (measured > allowed) | "under" (measured < allowed)
    severity: float = 0.5

    def __post_init__(self):
        if self.direction == "under" and self.allowed > 0 and self.measured < self.allowed:
            ratio = (self.allowed - self.measured) / max(self.allowed, 1e-9)
            self.severity = float(np.clip(ratio / 5.0, 0.0, 1.0))
        elif self.direction == "over" and self.allowed > 0 and self.measured > self.allowed:
            ratio = (self.measured - self.allowed) / max(self.allowed, 1e-9)
            self.severity = float(np.clip(ratio / 5.0, 0.0, 1.0))
        else:
            self.severity = 0.0
```
و در `rbta_enforcer.py` هنگامِ ساختِ نقضِ ENTROPY: `direction="under"` را صریح پاس بدهید.

**گزینه�ی ب (حداقلیِ فوری، اگر نمی‌خواهید امضای dataclass عوض شود):** در همان‌جایی که نقضِ ENTROPY ساخته می‌شود، severity را دستی محاسبه و override کنید:
```python
if entropy < bounds.entropy_floor:
    ratio = (bounds.entropy_floor - entropy) / max(bounds.entropy_floor, 1e-9)
    v = ConstraintViolation(module_id=module_id, bound_type="ENTROPY",
                             measured=entropy, allowed=bounds.entropy_floor)
    v.severity = float(np.clip(ratio / 5.0, 0.0, 1.0))
    violations.append(v)
```

**و مهم‌تر از هر دو — فیکسِ تست:** `test_entropy_floor_violation` باید `action` را هم assert کند:
```python
violations, action = enforcer.check_cycle(...)
assert action in (EnforcerAction.INTERRUPT, EnforcerAction.TERMINATE)
```
این تنها راهی است که این کلاس از رگرسیون در آینده دوباره جلوگیری می‌کند — چون همین الگو (assert فقط روی `violations`، نه روی `action`) ممکن است در تست‌های TIME/MEM/ENERGY هم تکرار شده باشد؛ ارزش دارد کل فایل را یک‌بار مرور کنید که هرجا `_` برای action استفاده شده، عمداً بوده نه فراموشی.

---

## بخش ۳: موارد از دورِ قبل که هنوز باز مانده‌اند (عمداً یا سهواً، نامشخص)

### F-04 (دورِ اول) — تکرارِ دستیِ hpm_spec/composition_tree — **هنوز حل نشده**
با `grep` تأیید شد: هنوز ۳ محلِ مجزا (`_run_perception_cycle` دو حالت + `step()`) ساختارِ درختِ ترکیب را از نو می‌سازند. این در لیستِ ۹ موردِ فیکس‌شده نبود؛ اگر قصد دارید ادامه دهید، این یک نامزدِ خوب برای دورِ بعدی refactor است (استخراج به یک متدِ مشترکِ `_build_composition_tree(mode)`).

### مسیرهای بررسی‌نشده از دورِ اول — هنوز باز
Consolidation/M3، خودِ فرمولِ Φ-IQ، FallbackController، و اسنادِ self-audit باقی‌مانده هنوز عمیق بررسی نشده‌اند (فقط اشاره‌ی سطحی داشتیم). اگر بخواهید، دورِ بعدی می‌تواند روی این‌ها متمرکز شود.

---

## بخش ۴: جمع‌بندیِ این دور

الگوی کلی این دور با دورِ قبل فرق دارد و این خودش خبرِ خوبی است: در دورِ اول، شکاف‌ها از جنسِ **«ادعا شده ولی هرگز پیاده نشده»** بودند (Attention مرده، MDIM دورزده‌شده). در این دور، فیکس‌های اصلی همه واقعی و به‌جا بودند — مشکلِ باقی‌مانده از جنسِ متفاوتی است: **یک انتزاعِ عمومی (severity برای «هر» نوع bound) که فقط برای یک جهتِ نقض (over) طراحی شده و بی‌صدا روی جهتِ دیگر (under) شکست می‌خورد.** این نوع باگ (یک‌طرفه‌بودنِ پنهانِ یک فرمولِ به‌ظاهر عمومی) دقیقاً همان‌جایی خطرناک است که تست‌ها هم آن را نمی‌گیرند، چون تستِ موجود فقط «شیء ساخته شد» را چک می‌کند نه «رفتارِ نهایی درست بود».

**توصیه‌ی مسیرِ بعدی:** پیشنهاد می‌کنم قبل از رفتن سراغِ بخش‌های نیامده، همین یک باگ (NEW-01) را با گزینه‌ی الف فیکس کنید و تستِ action را اضافه کنید — چون این دقیقاً یکی از پنج invariant بنیادینِ پروژه (A3) را در عمل خاموش می‌کند و ارزانتر است همین الان درست شود تا اینکه لایه‌های دیگر رویش بنا شوند.
