# Naive Bayes Classifier

יישום קטן שמדגים בניית מודל Naive Bayes וממשק שרת ו־UI פשוטים.

## התקנה

מומלץ לעבוד בסביבה וירטואלית. לאחר יצירת הסביבה התקינו את כל החבילות הדרושות.

```bash
pip install -r requirements.txt
pip install fastapi uvicorn streamlit pytest
```

חבילות עיקריות:
- `pandas`
- `scikit-learn`
- `fastapi` + `uvicorn`
- `streamlit`
- `pytest` לבדיקות

## הרצת השרת

לאחר התקנת התלויות ניתן להפעיל את השרת המקומי באמצעות:

```bash
uvicorn server:app --reload
```

כתובת ברירת המחדל תהיה `http://127.0.0.1:8000`.

## הרצת הבדיקות

להרצת הבדיקות המוגדרות בקובץ `comprehensive_tests.py`:

```bash
pytest comprehensive_tests.py -v
```

או לחלופין:

```bash
python comprehensive_tests.py
```

## דוגמה להרצת הסקריפט הראשי

הקובץ `main.py` מדגים טעינת נתונים והפעלת המודל. כדי להריץ אותו:

```bash
python main.py
```
במהלך ההפעלה יוצג תפריט לבחירת קובץ מתוך תיקיית `Data` או להזנת נתיב לקובץ CSV אחר.

## הפעלת ממשק Streamlit (אופציונלי)

ישנו קובץ `ui/user_interface.py` המאפשר אינטראקציה עם המודל בעזרת Streamlit:

```bash
streamlit run ui/user_interface.py
```
