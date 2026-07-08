import streamlit as st
import data_handler as dh
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
from train import get_all_scores
import joblib
from train import Models
import moduls as mls

def load_data():
    return pd.read_csv("bank-full.csv", sep=";")

df = load_data()

st.markdown(
    "<h1 style='text-align:center;color:green;'>Bank Marketing Natijasini Bashorat Qilish</h1>",
    unsafe_allow_html=True
)

img = Image.open("bank.jpg")
st.image(img, width=700)

df = pd.read_csv("bank-full.csv", sep=";")

show_data = st.sidebar.radio(
    "Datasetni ko'rishni xohlaysizmi?",
    ("Yo'q", "Ha")
)

if show_data == "Ha":

    st.subheader("Bank Marketing Ma'lumotlar To'plami")

    uz_df = df.copy()

    # Ustun nomlarini o'zbekchaga o'tkazish
    uz_df = uz_df.rename(columns={
        "age": "Yosh",
        "job": "Kasb",
        "marital": "Oilaviy_holat",
        "education": "Ta'lim",
        "default": "Qarz_to'lamagan",
        "balance": "Hisob_balansi",
        "housing": "Uy_krediti",
        "loan": "Shaxsiy_kredit",
        "contact": "Bog'lanish_turi",
        "day": "Kun",
        "month": "Oy",
        "duration": "Suhbat_davomiyligi",
        "campaign": "Kampaniya_aloqalari",
        "pdays": "Oxirgi_aloqadan_kun",
        "previous": "Oldingi_aloqalar",
        "poutcome": "Oldingi_natija",
        "y": "Depozit_ochdi"
    })

    # Qiymatlarni o'zbekchalashtirish

    uz_df.replace({
        "yes": "Ha",
        "no": "Yo'q",

        "married": "Turmush_qurgan",
        "single": "Bo'ydoq",
        "divorced": "Ajrashgan",

        "primary": "Boshlang'ich",
        "secondary": "O'rta",
        "tertiary": "Oliy",
        "unknown": "Noma'lum",

        "management": "Boshqaruv",
        "technician": "Texnik",
        "entrepreneur": "Tadbirkor",
        "blue-collar": "Ishchi",
        "retired": "Nafaqaxo'r",
        "admin.": "Administrator",
        "services": "Xizmat_ko'rsatish",
        "self-employed": "O'zini_band_qilgan",
        "unemployed": "Ishsiz",
        "housemaid": "Uy_xizmatchisi",
        "student": "Talaba",

        "cellular": "Mobil",
        "telephone": "Telefon",

        "success": "Muvaffaqiyatli",
        "failure": "Muvaffaqiyatsiz",
        "other": "Boshqa"
    }, inplace=True)

    st.dataframe(
        uz_df,
        use_container_width=True
    )

st.markdown("---")

show_dashboard = st.sidebar.radio(
    "Vizual dashboardni ko'rishni xohlaysizmi?",
    ("Yo'q","Ha")
)  

if show_dashboard=="Ha":

    st.header("📊 Vizual Dashboard")

jami=len(df)

yes=len(df[df["y"]=="yes"])

no=len(df[df["y"]=="no"])

balans=int(df["balance"].mean())

c1,c2,c3,c4=st.columns(4)

c1.metric("Jami mijoz",jami)

c2.metric("Depozit ochgan",yes)

c3.metric("Depozit ochmagan",no)

c4.metric("O'rtacha balans",balans)

# PIE CHAR ===================================
fig=px.pie(
    df,
    names="y",
    title="Depozit ochish holati"
)

st.plotly_chart(fig,use_container_width=True)

# HISTOGRAM ===================================
fig=px.histogram(
    df,
    x="age",
    color="y",
    title="Yosh bo'yicha mijozlar"
)

st.plotly_chart(fig,use_container_width=True)

# BAR CHAR ====================================
job = df.groupby("job").size().reset_index(name="Soni")

# Kasblarni o'zbekchaga tarjima qilish
job["job"] = job["job"].replace({
    "admin.": "Administrator",
    "blue-collar": "Ishchi",
    "entrepreneur": "Tadbirkor",
    "housemaid": "Uy xizmatchisi",
    "management": "Boshqaruv",
    "retired": "Nafaqaxo'r",
    "self-employed": "O'zini o'zi band qilgan",
    "services": "Xizmat ko'rsatish",
    "student": "Talaba",
    "technician": "Texnik",
    "unemployed": "Ishsiz",
    "unknown": "Noma'lum"
})

fig = px.bar(
    job,
    x="job",
    y="Soni",
    title="Kasblar bo'yicha mijozlar",
    labels={
        "job": "Kasblar",
        "Soni": "Mijozlar soni"
    }
)

st.plotly_chart(fig, use_container_width=True)


# SCATTER =================================
# Depozit holatini o'zbekchaga tarjima qilish
scatter_df = df.copy()

scatter_df["y"] = scatter_df["y"].replace({
    "yes": "Depozit ochgan",
    "no": "Depozit ochmagan"
})




fig = px.scatter(
    scatter_df,
    x="balance",
    y="duration",
    color="y",
    title="Balans va suhbat davomiyligi",
    labels={
        "balance": "Hisob balansi",
        "duration": "Suhbat davomiyligi (sekund)",
        "y": "Depozit holati"
    }
)

st.plotly_chart(fig, use_container_width=True)

# FILTER ====================================
st.subheader("Filtrlash")

# Kasblarni o'zbekchaga tarjima qilish
job_dict = {
    "management": "Boshqaruv",
    "technician": "Texnik",
    "entrepreneur": "Tadbirkor",
    "blue-collar": "Ishchi",
    "unknown": "Noma'lum",
    "retired": "Nafaqaxo'r",
    "admin.": "Administrator",
    "services": "Xizmat ko'rsatish",
    "self-employed": "O'zini o'zi band qilgan",
    "unemployed": "Ishsiz",
    "housemaid": "Uy xizmatchisi",
    "student": "Talaba",
    "businessman": "Biznesmen"
}

job_options = ["Barchasi"] + [job_dict.get(j, j) for j in df["job"].unique()]

selected_job = st.selectbox(
    "Kasbni tanlang",
    job_options
)

filtered_df = df.copy()

if selected_job != "Barchasi":
    reverse_dict = {v: k for k, v in job_dict.items()}
    selected_eng = reverse_dict[selected_job]
    filtered_df = filtered_df[filtered_df["job"] == selected_eng]

filtered_df = filtered_df.copy()
filtered_df["job"] = filtered_df["job"].replace(job_dict)

# Jadval nusxasi
table_df = filtered_df.copy()

# Ustun nomlarini o'zbekchaga o'zgartirish
table_df.rename(columns={
    "age": "Yosh",
    "job": "Kasb",
    "marital": "Oilaviy holati",
    "education": "Ta'lim",
    "default": "Kreditdan qarzdor",
    "balance": "Hisob balansi",
    "housing": "Uy krediti",
    "loan": "Shaxsiy kredit",
    "contact": "Aloqa turi",
    "day": "Kun",
    "month": "Oy",
    "duration": "Suhbat davomiyligi (sekund)",
    "campaign": "Kampaniya kontaktlari",
    "pdays": "Oxirgi kontaktdan o'tgan kunlar",
    "previous": "Oldingi kontaktlar",
    "poutcome": "Oldingi kampaniya natijasi",
    "y": "Depozit ochgan"
}, inplace=True)

# Kasblarni tarjima qilish
table_df["Kasb"] = table_df["Kasb"].replace({
    "management": "Boshqaruv",
    "technician": "Texnik",
    "entrepreneur": "Tadbirkor",
    "blue-collar": "Ishchi",
    "admin.": "Administrator",
    "services": "Xizmat ko'rsatish",
    "retired": "Nafaqaxo'r",
    "self-employed": "O'zini o'zi band qilgan",
    "student": "Talaba",
    "housemaid": "Uy xizmatchisi",
    "unemployed": "Ishsiz",
    "unknown": "Noma'lum"
})

# Oilaviy holat
table_df["Oilaviy holati"] = table_df["Oilaviy holati"].replace({
    "single": "Bo'ydoq/Turmush qurmagan",
    "married": "Turmush qurgan",
    "divorced": "Ajrashgan"
})

# Ta'lim
table_df["Ta'lim"] = table_df["Ta'lim"].replace({
    "primary": "Boshlang'ich",
    "secondary": "O'rta",
    "tertiary": "Oliy",
    "unknown": "Noma'lum"
})

# Ha / Yo'q qiymatlari
for col in ["Kreditdan qarzdor", "Uy krediti", "Shaxsiy kredit", "Depozit ochgan"]:
    table_df[col] = table_df[col].replace({
        "yes": "Ha",
        "no": "Yo'q"
    })

# Aloqa turi
table_df["Aloqa turi"] = table_df["Aloqa turi"].replace({
    "cellular": "Mobil telefon",
    "telephone": "Telefon",
    "unknown": "Noma'lum"
})

# Oylar
table_df["Oy"] = table_df["Oy"].replace({
    "jan": "Yanvar",
    "feb": "Fevral",
    "mar": "Mart",
    "apr": "Aprel",
    "may": "May",
    "jun": "Iyun",
    "jul": "Iyul",
    "aug": "Avgust",
    "sep": "Sentabr",
    "oct": "Oktabr",
    "nov": "Noyabr",
    "dec": "Dekabr"
})

# Oldingi kampaniya natijasi
table_df["Oldingi kampaniya natijasi"] = table_df["Oldingi kampaniya natijasi"].replace({
    "success": "Muvaffaqiyatli",
    "failure": "Muvaffaqiyatsiz",
    "other": "Boshqa",
    "unknown": "Noma'lum"
})

st.dataframe(table_df, use_container_width=True)

# ============================
# MODELLARNI KO'RISH BO'LIMI
# ============================

st.sidebar.markdown("---")

show_models = st.sidebar.radio(
    "Modellarni ko'rishni xohlaysizmi?",
    ("Yo'q", "Ha")
)

st.markdown("---")

if show_models == "Ha":

    st.markdown("---")
    st.subheader("Modellar natijalarini solishtirish")

    scores = get_all_scores()

    score_df = pd.DataFrame({
        "Model": list(scores.keys()),
        "Aniqlik (%)": list(scores.values())
    })

    st.dataframe(score_df, use_container_width=True)

    import plotly.express as px

    fig = px.bar(
        score_df,
        x="Model",
        y="Aniqlik (%)",
        text="Aniqlik (%)",
        title="Modellar aniqligini taqqoslash"
    )

    st.plotly_chart(fig, use_container_width=True)

predict_option = st.sidebar.radio(
    "Mijoz depozit ochishini tekshirish",
    ("Yo'q", "Ha")
)

try:

    if predict_option == "Ha":

        st.title("Depozit Bashorat Qilish")

        age = st.number_input("Yosh", 18, 100)

        balance = st.number_input("Hisob balansi")

        duration = st.number_input("Suhbat davomiyligi (sekund)")

        campaign = st.number_input("Kampaniya kontaktlari soni", 1)

        previous = st.number_input("Oldingi kontaktlar soni", 0)

        housing = st.selectbox(
            "Uy krediti bormi?",
            ("yes", "no")
        )

        loan = st.selectbox(
            "Shaxsiy kredit bormi?",
            ("yes", "no")
        )

        housing = 1 if housing == "yes" else 0
        loan = 1 if loan == "yes" else 0

        x = pd.DataFrame({
            "age":[age],
            "balance":[balance],
            "duration":[duration],
            "campaign":[campaign],
            "previous":[previous],
            "housing":[housing],
            "loan":[loan]
        })

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = dh.preprocess("bank-full.csv")

        x_scaled = scaler.transform(x)
        model = joblib.load("xgb.joblib")
        predictions = np.array([
            model.predict(x_scaled)
            for model in mls.models()
        ])

        predict = st.button("Bashorat qilish")

        if predict:

            if predictions.mean() >= 0.5:
                st.success("Mijoz omonat ochishi ehtimoli yuqori.")
            else:
                st.error("Mijoz omonat ochmasligi mumkin.")

except:
    st.error("Ma'lumotlarni to'g'ri kiriting!")

about = st.sidebar.button("Muallif")

if about:

    st.markdown(
        """
        <h2 style='text-align:center; color:green;'>
        Muallif haqida
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Loyiha asoschisi

        **Ism va familiya:** Munisaxon Qurbonova

        **Elektron pochta:** munisakurbanova14@gmail.com

        **Loyiha mavzusi:** Bank Marketing Natijasini Bashorat Qilish

        **Texnologiyalar:**
        - Python
        - Streamlit
        - Pandas
        - Scikit-Learn
        - XGBoost
        - Machine Learning

        ---
        Ushbu loyiha marketing kampaniyasidan keyin mijoz muddatli depozit ochadimi yoki yo'qligini bashorat qilish uchun ishlab chiqilgan.
        """
    )

    st.success("E'tiboringiz uchun rahmat!")