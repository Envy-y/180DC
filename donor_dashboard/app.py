import streamlit as st
import pandas as pd
import plotly.express as px
from clean_data import clean_data


st.set_page_config(page_title="Donor Dashboard", layout="wide")  

st.markdown("""
<style>

.st-key-cont {
            background-color: #ffffff;
            }

.st-key-cont2 {
            background-color: #ffffff;
            }
.st-key-cont3 {
            background-color: #ffffff;
            }
.st-key-cont4 {
            background-color: #ffffff;
            }

/* Title + headers */
.report-title {
  font-size: 42px !important;
  color: #0f172a;
  font-weight: 800;
  margin-top: 10px;
  margin-bottom: 4px;
}
.report-header {
  font-size: 26px !important;
  color: var(--muted);
  border-bottom: 2px solid rgba(0,0,0,0.08);
  padding-bottom: 8px;
  font-weight: 700;
  margin-top: 18px;
  margin-bottom: 12px;
}
.report-subheader {
  font-size: 20px !important;
  color: var(--primary);
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .02em;
  margin-top: 14px;
  margin-bottom: 8px;
}
.report-subsubheader {
  font-size: 16px !important;
  color: var(--muted);
  font-weight: 700;
  margin-top: 10px;
  margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# Load your data
@st.cache_data
def load_data():
    # replace with your real dataset
    return pd.read_csv("donor_dashboard/WVS7.csv", delimiter = ";")

fd_new = load_data()
fd_cleaned = clean_data(fd_new)

st.markdown("<div class='report-title'>Donor Market Insights: Quantitative Analysis using WVS Wave 7</div>", unsafe_allow_html=True)

# Page title and description
col1, col2= st.columns([3,3])  # 2 equal columns
with col1:
    st.image("donor_dashboard/compassion_logo.png", width=250)
with col2:
    st.image("donor_dashboard/180DC_logo.png", width=150)

# Dashboard summary section
st.markdown("<div class='report-header'>Dashboard Summary</div>", unsafe_allow_html=True)
st.markdown("""
This dashboard provides a data-driven perspective on India’s donor landscape using the World Values Survey (WVS) Wave 7 (India sample). It brings together a wide set of demographic attributes (age, gender, education, income etc.) and attitudinal measures (trust & confidence) to explore how these factors relate to donor behaviour.
            
The purpose of the dashboard is to present a series of interactive visualisations that allow users to examine donor potential across different segments and characteristics. These visual insights are then complemented by a logistic regression model that estimates the probability of an individual becoming a donor, based on their demographic and attitudinal profile.
            
Ultimately, the dashboard highlights key donor segments and behavioural drivers that Compassion can leverage in shaping its India market entry strategy.
""")

# First chart: Potential Donor by Demographic

# Sidebar controls


with st.container(border=True, key = "cont"):
    st.markdown("<div class='report-subheader'>1. Potential Donor by Demographic</div>", unsafe_allow_html=True)
    x_var = st.selectbox("Choose a variable:", 
                     ["Age Group", "Gender", "Education Level", 
                      "Income Level", "Religion","Household Size", "Marital Status"])
    def plot_first_chart(x_var):
        # ----- orders -----
        age_order = ["Gen Z", "Millennial", "Gen X", "Boomer", "Silent"]
        edu_order = [
            "Primary or below", "Secondary", "Post-secondary / Diploma",
            "Bachelor", "Postgraduate", "No Response"
        ]
        income_order = [  
            "Low Income", "Middle Income", "High Income", "No Response"
        ]
        household_order = [
            "Small (1-2)", "Medium (3-5)", "Large (6-8)", "Very large (9+)", "No Response"
        ]
        marital_order = [  
            "Single", "Married", "Living together", "Divorced", "Widowed", "No Response"
        ]

        orders = {
            "Age Group": age_order,
            "Education Level": edu_order,
            "Income Level": income_order,
            "Household Size": household_order,
            "Marital Status": marital_order,
        }

        # Compute counts and proportions
        df = (
            fd_cleaned.groupby([x_var, "Potential_Donor_Str"])
            .size()
            .reset_index(name="count")
        )

        # Normalize labels to avoid 'No response' / 'No Response ' issues
        df[x_var] = (
            df[x_var]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace("no response", "No Response", case=False, regex=False)
        )

        df["pct"] = df["count"] / df.groupby(x_var)["count"].transform("sum")

        mode = st.radio("Show values as:", ["Count", "Percentage"])
        hide_noresp = st.checkbox("Hide 'No Response'", value=True)

        df_plot = df.copy()
        if hide_noresp:
            df_plot = df_plot[df_plot[x_var] != "No Response"].copy()
            # If the original column was categorical, drop the unused category
            if isinstance(fd_cleaned[x_var], pd.CategoricalDtype):
                try:
                    df_plot[x_var] = df_plot[x_var].cat.remove_categories(["No Response"])
                except Exception:
                    pass

        # Choose order, then keep only categories that are present after filtering
        if x_var in orders:
            base_order = orders[x_var]
            present = df_plot[x_var].unique().tolist()
            # preserve order but keep only present levels
            category_order = [c for c in base_order if c in present]
        else:
            category_order = df_plot.groupby(x_var)["count"].sum().sort_values(ascending = False).index.tolist()

        # Y and text
        if mode == "Count":
            y_col = "count"
            display = "count"
            y_title = "Respondent Count"
        else:
            y_col = "pct"
            df_plot["pct_label"] = (df_plot["pct"] * 100).round(0).astype(int).astype(str) + "%"
            display = "pct_label"
            y_title = "Proportion of respondents"

        fig = px.bar(
            df_plot,
            x=x_var,
            y=y_col,
            color="Potential_Donor_Str",
            text=display,
            category_orders={x_var: category_order},
            color_discrete_map={"Yes": "#55b441", "No": "lightgrey"},
            barmode="stack",
            title=f"Potential Donor by {x_var}",
            labels={"Potential_Donor_Str": "Potential Donor", "pct_label": "Percentage"},
            hover_data={"pct": False},
        )

        fig.update_traces(textposition="inside", textangle=0)
        if mode == "Percentage":
            fig.update_yaxes(tickformat=".0%", range=[0, 1])

        fig.update_layout(yaxis_title=y_title)
        return fig
    st.plotly_chart(plot_first_chart(x_var), use_container_width=True)
    st.markdown("<div class='report-subsubheader'>Key Insights</div>", unsafe_allow_html=True)
    st.write("""
            - **Age:** Millennials and the Silent Generation show strong donor potential.  
            - **Gender:** Males are more likely to donate (**78%**) compared to females (**62%**).  
            - **Education:** Donor potential rises consistently with education — peaking at **postgraduate (~78%)**.  
            - **Income:** Fairly stable across low, middle, and high incomes (**70–72%**), with an anomaly in the “No Response” group (**~79%**), possibly reflecting ultra-high net worth individuals reluctant to disclose.  
            - **Religion:** High donor potential across most groups, with **Christians (81%)** and **Muslims (76%)** at the top.  
            - **Household Size:** Very large households (9+ members) exhibit the highest donor potential at (**~79%**).
            - **Marital Status:** Widowed individuals show low donor potential at (**51%**).
            """)


with st.container(border=True, key = "cont2"):
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

    st.markdown("<div class='report-subsubheader'>Key Insights</div>", unsafe_allow_html=True)
    st.write("""
    - **Uttar Pradesh**: Exceptionally strong donor potential — **Rural 85%** and **Urban 89%**.  
    - **Delhi**: High donor likelihood overall, especially **Urban at 80%**.  
    - **Bihar**: Strong donor potential in both **Rural (75%)** and **Urban (81%)**.  
    - **West Bengal**: The only state with a split profile — **Rural donors only 45%**, while **Urban improves to 67%**.  
    - **Maharashtra**: Mixed results — **Rural 69%** donor potential, but **Urban drops to 62%**.  
    - **Andhra Pradesh**: **Rural areas weaker (59%)**, but **Urban shows high donor potential (83%)**.  
    - **Punjab**: Moderate donor potential — **Urban 62%** vs **Rural 67%**.  
    - **Haryana**: Relatively balanced with **Rural 67%** and **Urban 62%**.  
    - **Overall:** Donor potential is consistently stronger in **Urban areas**, with Uttar Pradesh, Delhi, and Bihar leading as donor hotspots. West Bengal is the main outlier with low rural donor engagement.
    """)


with st.container(border=True, key = "cont3"):
# Third Chart: Trust and Closeness
    st.markdown("<div class='report-subheader'>3. Trust & Closeness to the World </div>", unsafe_allow_html=True)

    def plot_third_chart():
        import plotly.graph_objects as go

        # Build low/high probs and uplift
        vars_map = {
            "Closeness_World": "Closeness to the world",
            "Trust_Charities": "Trust in charitable orgs",
            "Trust_UN": "Trust in UN",
            "Trust_Churches": "Trust in churches",
            "Trust_OtherReligion": "Trust people from other religion",
        }
        rows = []
        for col, lab in vars_map.items():
            grp = fd_cleaned.groupby(col)["Potential_Donor"].mean()
            low  = grp.loc[[1,2]].mean() 
            high = grp.loc[[3,4]].mean() 
            rows.append({"Variable": lab, "Low": low, "High": high, "Uplift": high-low})
        df = pd.DataFrame(rows).dropna()

        # Sort by uplift (largest at top)
        y = df["Variable"]

        # Build Plotly figure
        fig = go.Figure()

        # Add dumbbell lines
        for i, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Low"], row["High"]],
                y=[row["Variable"], row["Variable"]],
                mode="lines",
                line=dict(color="lightgray", width=3),
                showlegend=False,
                hoverinfo="skip"
            ))

        # Add Low points
        fig.add_trace(go.Scatter(
            x=df["Low"], y=y,
            mode="markers+text",
            marker=dict(color="gray", size=10),
            name="Low (Category 1-2)",
            text=[f"{v:.0%}" for v in df["Low"]],
            textposition="middle left",
            textfont=dict(color="dimgray", size=10),
            hovertemplate="%{y}<br>Low: %{x:.0%}<extra></extra>"
        ))

        # Add High points
        fig.add_trace(go.Scatter(
            x=df["High"], y=y,
            mode="markers+text",
            marker=dict(color="#55b441", size=10),
            name="High (Category 3-4)",
            text=[f"{v:.0%}" for v in df["High"]],
            textposition="middle right",
            textfont=dict(color="#2e7031", size=10),
            hovertemplate="%{y}<br>High: %{x:.0%}<extra></extra>"
        ))

        # Layout
        fig.update_layout(
            title="Impact on Donor Probability (Low vs High)",
            xaxis=dict(
                range=[0.6,0.8],
                tickvals= [0.6,0.7,0.8],
                ticktext=["60%","70%","80%"],
                title="Donor probability"
            ),
            yaxis=dict(title=""),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(title="", orientation="h", y=1.1, x=0.3)
        )

        st.plotly_chart(fig, use_container_width=True)

    plot_third_chart()

    st.markdown("<div class='report-subsubheader'>Key Insights</div>", unsafe_allow_html=True)

    st.write("""
    - Across all five trust/closeness variables, **higher trust (categories 3–4) is associated with a greater likelihood of being a potential donor**.  
    - **Trust in the UN** shows the lowest baseline (63%) but still rises meaningfully to 75% at high trust.  
    - Overall, the data highlights that **trust and global connectedness are positive drivers of donor potential**, with the strongest effect seen in world-mindedness.
    """)



with st.container(border=True, key = "cont4"):
    st.markdown("<div class='report-subheader'>4. Build Your Own Profile </div>", unsafe_allow_html=True)

    st.markdown("This section lets you **build your own profile** and see how likely someone with those characteristics is to become a donor. Behind the scenes, we use a statistical method called *logistic regression*. This model looks at patterns in past data — such as age group, education, income, or household size — and estimates the probability that someone with a similar profile would donate. Think of it as drawing on our entire dataset and say, *“people like this are usually donors about X% of the time.”*")

    c1,c2,c3,c4 = st.columns([1,1,1,1]) 
    c5,c6,c7,c8 = st.columns([1,1,1,1])

    with c1:
        age = st.selectbox("Age Group", 
                        ["Gen Z", "Millennial", "Gen X", "Boomer", "Silent"])
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with c3:
        edu = st.selectbox("Education Level", 
                        ["Primary or below", "Secondary", "Post-secondary / Diploma", "Bachelor", "Postgraduate"])
    with c4:
        income = st.selectbox("Income Level", 
                            ["Low Income", "Middle Income", "High Income"])
    with c5:
        religion = st.selectbox("Religion", 
                                ["Hindu", "Muslim", "Christian", "Buddhist", "Other (e.g. Sikh)"])
    with c6:
        hh_size = st.selectbox("Household Size", 
                            ["Small (1-2)", "Medium (3-5)", "Large (6-8)", "Very large (9+)"])
    with c7:
        marital = st.selectbox("Marital Status", ["Single", "Married", "Living together","Widowed"])
    with c8:
        urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])

    def train_reg():
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        X = fd_cleaned[[
            "Age Group", "Gender", "Education Level",
            "Income Level", "Religion", "Household Size",
            "Marital Status", "Urban/Rural"
        ]]
        y = (fd_cleaned["Potential_Donor_Str"] == "Yes").astype(int)
        categorical_features = X.columns.tolist()
        categorical_transformer = OneHotEncoder(drop="first")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Logistic regression pipeline
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        clf.fit(X, y)
        return clf

    @st.cache_data
    def get_model():
        return train_reg()

    model = get_model()

    # Collect inputs
    input_dict = {
        "Age Group": age,
        "Gender": gender,
        "Education Level": edu,
        "Income Level": income,
        "Religion": religion,
        "Household Size": hh_size,
        "Marital Status": marital,
        "Urban/Rural": urban_rural,
    }

    # Convert to DataFrame for pipeline
    input_df = pd.DataFrame([input_dict])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]  

    def res_card(title, value, prob, color="#55b441"):
        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #f5f5f5;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="margin:0; font-size:20px; color:#333;">{title}</h4>
                <p style="margin:5px 0 10px 0; font-size:28px; font-weight:bold; color:{color};">
                    {value}
                </p>
                <!-- Progress bar container -->
                <div style="background-color:#e0e0e0; border-radius:8px; height:14px; width:100%;">
                    <div style="
                        background-color:{color};
                        width:{prob*100}%;
                        height:100%;
                        border-radius:8px;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    res_card("Predicted Donor Probability", f"{prob*100:.1f}%", prob)

    st.markdown("<div class='report-subsubheader'>Limitations</div>", unsafe_allow_html=True)
    st.markdown("""It’s important to remember that these predictions are **estimates, not guarantees**. While logistic regression is useful for spotting broad patterns, it cannot capture every nuance of individual behaviour. Limitations include:
    - **Simplified relationships** — the model assumes each factor influences donor likelihood in a linear way, which may not reflect reality.  
    - **Weak predictors** — some inputs may have little or no real impact, even if included in the model.  
    - **Missing factors** — personal values, cultural context, and unique motivations are not represented in the data.  
    - **Uncertainty** — the probability shown comes with a margin of error; small changes in data can shift results.  
    - **Not individualised** — the output reflects group trends, not a precise prediction for one person. 
    - **Small subgroups** — if very few people in the dataset share a specific profile (e.g. **Christians**), the model’s estimates for that group will be less reliable.   """ )



