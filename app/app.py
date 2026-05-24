import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Sales Prediction Dashboard",
    layout="wide"
)

# ==========================================
# TITLE SECTION
# ==========================================

st.title("📊 AI Sales Prediction Dashboard")

st.write(
    "Sales Prediction and AI-Based Business Recommendation System Using Python"
)

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/Advertising Budget and Sales.csv")

# ==========================================
# SIDEBAR MENU
# ==========================================

st.sidebar.title("Dashboard Menu")

menu = st.sidebar.selectbox(
    "Select Option",
    [
        "Dataset",
        "Analytics",
        "Prediction"
    ]
)

# ==========================================
# DATASET SECTION
# ==========================================

if menu == "Dataset":

    st.subheader("📁 Dataset Preview")

    st.dataframe(data)

    st.subheader("📈 Dataset Statistics")

    st.write(data.describe())

# ==========================================
# ANALYTICS SECTION
# ==========================================

elif menu == "Analytics":

    st.subheader("📊 Sales Analytics Dashboard")

    # KPI CARDS

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Sales",
        f"{data['Sales ($)'].mean():.2f}"
    )

    col2.metric(
        "Maximum Sales",
        f"{data['Sales ($)'].max():.2f}"
    )

    col3.metric(
        "Minimum Sales",
        f"{data['Sales ($)'].min():.2f}"
    )

    st.write("---")

    # ==========================================
    # TV ADS VS SALES
    # ==========================================

    st.subheader("TV Advertising vs Sales")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.scatter(
        data['TV Ad Budget ($)'],
        data['Sales ($)']
    )

    ax1.set_xlabel("TV Advertising Budget")

    ax1.set_ylabel("Sales")

    ax1.set_title("TV Advertising Impact")

    st.pyplot(fig1)

    # ==========================================
    # RADIO ADS VS SALES
    # ==========================================

    st.subheader("Radio Advertising vs Sales")

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.scatter(
        data['Radio Ad Budget ($)'],
        data['Sales ($)']
    )

    ax2.set_xlabel("Radio Advertising Budget")

    ax2.set_ylabel("Sales")

    ax2.set_title("Radio Advertising Impact")

    st.pyplot(fig2)

    # ==========================================
    # NEWSPAPER ADS VS SALES
    # ==========================================

    st.subheader("Newspaper Advertising vs Sales")

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.scatter(
        data['Newspaper Ad Budget ($)'],
        data['Sales ($)']
    )

    ax3.set_xlabel("Newspaper Advertising Budget")

    ax3.set_ylabel("Sales")

    ax3.set_title("Newspaper Advertising Impact")

    st.pyplot(fig3)

# ==========================================
# PREDICTION SECTION
# ==========================================

elif menu == "Prediction":

    st.subheader("🔮 Predict Future Sales")

    # ==========================================
    # MACHINE LEARNING MODEL
    # ==========================================

    X = data[
        [
            'TV Ad Budget ($)',
            'Radio Ad Budget ($)',
            'Newspaper Ad Budget ($)'
        ]
    ]

    y = data['Sales ($)']

    model = LinearRegression()

    model.fit(X, y)

    # ==========================================
    # USER INPUTS
    # ==========================================

    st.write("Enter Advertising Budgets")

    tv = st.slider(
        "TV Advertising Budget",
        0,
        300,
        150
    )

    radio = st.slider(
        "Radio Advertising Budget",
        0,
        50,
        25
    )

    newspaper = st.slider(
        "Newspaper Advertising Budget",
        0,
        100,
        50
    )

    # ==========================================
    # CREATE INPUT DATAFRAME
    # ==========================================

    input_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=[
            'TV Ad Budget ($)',
            'Radio Ad Budget ($)',
            'Newspaper Ad Budget ($)'
        ]
    )

    # ==========================================
    # SALES PREDICTION
    # ==========================================

    prediction = model.predict(input_data)

    st.subheader("📈 Predicted Sales")

    st.success(
        f"Predicted Sales: {prediction[0]:.2f}"
    )

    # ==========================================
    # AI BUSINESS RECOMMENDATIONS
    # ==========================================

    st.subheader("🤖 AI Business Recommendations")

    if prediction[0] < 10:

        st.warning("Increase marketing budget.")

        st.warning("Improve customer awareness.")

        st.warning("Offer attractive discounts.")

    elif prediction[0] < 20:

        st.info("Focus on digital marketing.")

        st.info("Launch seasonal promotions.")

        st.info("Improve customer engagement.")

    else:

        st.success("Increase inventory stock.")

        st.success("Expand marketing campaigns.")

        st.success("Focus on high-performing products.")

        st.success("Explore business expansion opportunities.")

    # ==========================================
    # DOWNLOAD REPORT
    # ==========================================

    report = pd.DataFrame({
        "TV Budget": [tv],
        "Radio Budget": [radio],
        "Newspaper Budget": [newspaper],
        "Predicted Sales": [prediction[0]]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="sales_prediction_report.csv",
        mime="text/csv"
    )

# ==========================================
# FOOTER
# ==========================================

st.write("---")

st.write("MBA Mini Project")

st.write(
    "Sales Prediction and AI-Based Business Recommendation System Using Python"
)
