import streamlit as st
import pandas as pd
import plotly.express as px
from clean_data import clean_data

st.markdown(
    """
    <style>
    .report-title {
        font-size: 42px !important;
        color: #000000;
        text-align: left;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .report-header {
        font-size: 30px !important;
        color: #414042;
        border-bottom: 2px solid #414042;
        padding-bottom: 4px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .report-subheader {
        font-size: 24px !important;
        color: #55b441;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(page_title="Donor Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load your data
@st.cache_data
def load_data():
    # replace with your real dataset
    return pd.read_csv("WVS7.csv", delimiter = ";")

fd_new = load_data()
fd_cleaned = clean_data(fd_new)

st.markdown("<div class='report-title'>Donor Market Insights: Quantitative Analysis using WVS Wave 7</div>", unsafe_allow_html=True)

# Page title and description
col1, col2= st.columns([3,3])  # 2 equal columns
with col1:
    st.image("compassion_logo.png", width=250)
with col2:
    st.image("180DC_logo.png", width=150)

# Dashboard summary section
st.markdown("<div class='report-header'>Dashboard Summary</div>", unsafe_allow_html=True)
st.markdown("""
This dashboard provides a data-driven perspective on India’s donor landscape using the World Values Survey (WVS) Wave 7 (India sample). It brings together a wide set of demographic attributes (age, gender, education, income etc.) and attitudinal measures (trust & confidence) to explore how these factors relate to donor behaviour.
            
The purpose of the dashboard is to present a series of interactive visualisations that allow users to examine donor potential across different segments and characteristics. These visual insights are then complemented by a logistic regression model that estimates the probability of an individual becoming a donor, based on their demographic and attitudinal profile.
            
Ultimately, the dashboard highlights key donor segments and behavioural drivers that Compassion can leverage in shaping its India market entry strategy.
""")

# First chart: Potential Donor by Demographic
st.markdown("<div class='report-subheader'>1. Potential Donor by Demographic</div>", unsafe_allow_html=True)
# Sidebar controls
x_var = st.selectbox("Choose a variable:", 
                     ["Age Group", "Gender", "Education Level", 
                      "Income Level", "Religion"])

def plot_first_chart(x_var):
        # Define category orders for specific variables
    age_order = ["Gen Z", "Millennial", "Gen X", "Boomer", "Silent"]
    edu_order = [
        "Primary or below",
        "Secondary",
        "Post-secondary / Diploma",
        "Bachelor",
        "Postgraduate",
        "No Response"
    ]
    income_order = [
        "Low Income",      
        "Middle Income",   
        "High Income",     
        "No Response"
    ]

    orders = {
        "Age Group": age_order,
        "Education Level": edu_order,
        "Income Level": income_order
    }

    if x_var in orders:
        category_order = orders[x_var]
    else:
        # sort by counts descending
        category_order = (
            fd_cleaned[x_var].value_counts().index.tolist()
        )


    # Compute counts and proportions
    df = (fd_cleaned.groupby([x_var, "Potential_Donor_Str"])
                .size()
                .reset_index(name="count"))
    df["pct"] = df["count"] / df.groupby(x_var)["count"].transform("sum")

    # Sidebar controls
    mode = st.radio("Show values as:", ["Count", "Percentage"])
    hide_noresp = st.checkbox("Hide 'No Response'", value=True)

    # Filter if checkbox ticked
    df_plot = df.copy()
    if hide_noresp:
        df_plot = df_plot[df_plot[x_var] != "No Response"]

    # Pick y and text depending on mode
    if mode == "Count":
        y_col = "count"
        y_title = "Respondent Count"
    else:
        y_title = "Proportion of respondents"
        df_plot["pct_label"] = (df_plot["pct"] * 100).round(0).astype(int).astype(str) + "%"
        y_col = "pct"


    display = "count" if mode == "Count" else "pct_label"

    fig = px.bar(
        df_plot,
        x=x_var,
        y=y_col,
        color="Potential_Donor_Str",
        text=display,
        category_orders={x_var: category_order},
        color_discrete_map={"Yes":"#55b441", "No":"lightgrey", "No Response":"#cccccc"},
        barmode="stack",
        title=f"Potential Donor by {x_var}",
        labels={"Potential_Donor_Str": "Potential Donor", "pct_label": "Percentage"},
        hover_data={"pct": False}
    )
    fig.update_traces(
        textposition="inside",   # options: 'inside', 'outside', 'auto', 'none'
        textangle=0              # force horizontal
    )
    # If percentage, force y-axis to 0–1 with percent ticks
    if mode == "Percentage":
        fig.update_yaxes(tickformat=".0%", range=[0,1])

    fig.update_layout(yaxis_title=y_title)

    st.plotly_chart(fig, use_container_width=True)

plot_first_chart(x_var)


# Second Chart: Geographic Distribution of Respondents
st.markdown("<div class='report-subheader'>2. Geograpic Distribution of Potential Donor</div>", unsafe_allow_html=True)
st.markdown("""
This treemap visualises the geographic distribution of WVS respondents across India, segmented by state and urban/rural status. The size of each rectangle represents the number of respondents in that category, while the color indicates their donor status (light green= potential donor, grey = non-donor).

You can click on any rectangle to zoom in and explore the breakdown of donor status within that segment. For example, clicking on "Delhi" will show the split between urban and rural respondents in Delhi, and further clicking on "Urban" will reveal the counts of potential donors vs non-donors among urban Delhi respondents.
            """)

def plot_second_chart():
    counts = (
    fd_cleaned.groupby(["State", "Urban/Rural", "Potential_Donor_Str"])
      .size()
      .reset_index(name="count")    
    )

    fig = px.treemap(
    counts,
    path=["State", "Urban/Rural", "Potential_Donor_Str"],
    values="count",
    color="Potential_Donor_Str",
    color_discrete_map={'(?)':'Green',"Yes":"#55b441","No":"lightgrey"},
    labels={"Potential_Donor_Str": "Potential Donor"}
    )
    fig.update_layout(title="Donor Status by State and Urban/Rural")
    fig.update_traces(
        root_color="#eeeeee"
    )

    fig.update_traces(
    hovertemplate=(
        "<b>%{parent}</b> → %{label}<br>"
        "Donor Status: %{label}<br>"
        "Count: %{value}<br>"
        "Share of parent: %{percentParent:.1%}<br>"
        "<extra></extra>"
    )   
    )
    
    fig.update_traces(
        texttemplate="%{label}<br>%{percentParent:.0%}",  # label + percent
        textinfo="label+text"
    )
 
    fig.update_layout(uniformtext=dict(minsize=10))



    st.plotly_chart(fig, use_container_width=True)
plot_second_chart()

# Third Chart: Trust and Closeness
st.markdown("<div class='report-subheader'>3. Trust & Closness to the World </div>", unsafe_allow_html=True)



