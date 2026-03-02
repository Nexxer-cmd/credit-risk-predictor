import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Set page config 
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers with gradient text for a premium feel */
    h1 {
        color: #E2E8F0;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: -webkit-linear-gradient(45deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #E2E8F0;
        font-weight: 600;
        letter-spacing: -0.015em;
    }
    
    /* Accent color for sliders/inputs */
    div[data-baseweb="slider"] {
        accent-color: #3b82f6;
    }
    
    /* Metrics box styling */
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    /* Subtle borders for containers */
    .stContainer {
        background-color: #1A1D24;
        border-radius: 12px;
        padding: 30px;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stContainer:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    }
    
    /* Styling the primary buttons */
    .stButton>button[kind="primary"] {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    /* Styling secondary buttons */
    .stButton>button[kind="secondary"] {
        width: 100%;
        background-color: #2D3748;
        color: #E2E8F0;
        font-weight: 600;
        border: 1px solid #4A5568;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background-color: #4A5568;
        border-color: #718096;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.02) 100%);
        border-left: 4px solid #3b82f6;
        padding: 15px 20px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 25px;
        color: #E2E8F0;
        font-size: 1rem;
    }
    
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        background-color: #11141A;
        border-right: 1px solid #2D3748;
    }
    
    /* Radio button active state highlight */
    div.row-widget.stRadio > div{
        flex-direction:column;
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_data():
    try:
        # Load a sample for data exploration features
        df = pd.read_csv("cs-training.csv", nrows=10000)
        df = df.rename(columns={'Unnamed: 0': 'ID'})
        return df
    except:
        return pd.DataFrame()

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Risk Predictor")
    
    st.markdown("### Navigation Menu")
    page = st.radio(
        "",
        ["Risk Assessment System", "Data Insights Dashboard", "System Architecture"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.85rem; color: #A0AEC0;'>
            <b>Objective:</b> Assess the likelihood of a borrower defaulting on a loan within the next two years using Machine Learning.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Powered by HistGradientBoostingClassifier")


# --- Page Routing Logic ---

if page == "Risk Assessment System":
    st.title("Credit Default Risk Analysis", anchor=False)
    st.markdown("<div class='info-box'>Welcome. Please enter the applicant's financial profile below to generate an AI-powered risk assessment.</div>", unsafe_allow_html=True)

    # Initialize session state for wizard steps
    if 'step' not in st.session_state:
        st.session_state.step = 1

    def next_step():
        st.session_state.step += 1

    def prev_step():
        st.session_state.step -= 1

    # Progress bar with color
    st.markdown(f"<p style='color: #60a5fa; font-weight: 600; margin-bottom: 5px;'>Step {st.session_state.step} of 3</p>", unsafe_allow_html=True)
    st.progress(st.session_state.step / 3.0)

    # Form variables
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        if st.session_state.step == 1:
            st.subheader("Personal & Income Details", anchor=False)
            st.markdown("<p style='color: #A0AEC0;'>Demographic and income information.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_left, col_right = st.columns(2, gap="large")
            with col_left:
                st.session_state.form_data['age'] = st.slider(
                    "Borrower Age", min_value=18, max_value=100, 
                    value=st.session_state.form_data.get('age', 35), 
                    help="Age of the borrower in years."
                )
                st.session_state.form_data['dependents'] = st.number_input(
                    "Number of Dependents", min_value=0, max_value=20, 
                    value=st.session_state.form_data.get('dependents', 1), step=1,
                    help="Number of dependents in the household."
                )
            with col_right:
                st.session_state.form_data['monthly_income'] = st.number_input(
                    "Monthly Income ($)", min_value=0, 
                    value=st.session_state.form_data.get('monthly_income', 5000), step=500,
                    help="Monthly gross income of the borrower."
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            _, btn_col = st.columns([3, 1])
            with btn_col:
                st.button("Next Step ⭢", on_click=next_step, type="primary", use_container_width=True)

        elif st.session_state.step == 2:
            st.subheader("Credit Utilization", anchor=False)
            st.markdown("<p style='color: #A0AEC0;'>Active credit limits and existing debts.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_left, col_right = st.columns(2, gap="large")
            with col_left:
                st.session_state.form_data['revolving_utilization'] = st.slider(
                    "Revolving Utilization", min_value=0.0, max_value=2.0, 
                    value=st.session_state.form_data.get('revolving_utilization', 0.3), step=0.05,
                    help="Total balance on credit cards and personal lines divided by total credit limit."
                )
                st.session_state.form_data['debt_ratio'] = st.number_input(
                    "Debt Ratio", min_value=0.0, max_value=10.0, 
                    value=st.session_state.form_data.get('debt_ratio', 0.4), step=0.1,
                    help="Monthly debt payments, alimony, living costs divided by monthly gross income."
                )
            with col_right:
                st.session_state.form_data['open_lines'] = st.number_input(
                    "Open Credit Lines/Loans", min_value=0, max_value=50, 
                    value=st.session_state.form_data.get('open_lines', 5), step=1,
                    help="Number of open loans and active lines of credit."
                )
                st.session_state.form_data['real_estate_lines'] = st.number_input(
                    "Real Estate Loans", min_value=0, max_value=15, 
                    value=st.session_state.form_data.get('real_estate_lines', 1), step=1,
                    help="Number of mortgage and real estate loans."
                )

            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, _, btn_col2 = st.columns([1, 2, 1])
            with btn_col1:
                st.button("⭠ Back", on_click=prev_step, use_container_width=True)
            with btn_col2:
                st.button("Next Step ⭢", on_click=next_step, type="primary", use_container_width=True)

        elif st.session_state.step == 3:
            st.subheader("Delinquency History", anchor=False)
            st.markdown("<p style='color: #A0AEC0;'>Number of times the applicant has been past due on obligations.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                st.session_state.form_data['past_due_30_59'] = st.number_input(
                    "30-59 Days Late", min_value=0, max_value=20, 
                    value=st.session_state.form_data.get('past_due_30_59', 0), step=1
                )
            with col2:
                st.session_state.form_data['past_due_60_89'] = st.number_input(
                    "60-89 Days Late", min_value=0, max_value=20, 
                    value=st.session_state.form_data.get('past_due_60_89', 0), step=1
                )
            with col3:
                st.session_state.form_data['past_due_90_plus'] = st.number_input(
                    "90+ Days Late", min_value=0, max_value=20, 
                    value=st.session_state.form_data.get('past_due_90_plus', 0), step=1
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, _, btn_col2 = st.columns([1, 1.5, 1.5])
            with btn_col1:
                st.button("⭠ Back", on_click=prev_step, use_container_width=True)
            with btn_col2:
                analyze_pressed = st.button("Analyze Risk Profile", type="primary", use_container_width=True)
                
                if analyze_pressed:
                    st.session_state.analyze_trigger = True

    # --- Results Section ---
    if getattr(st.session_state, 'analyze_trigger', False):
        fd = st.session_state.form_data
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': fd['revolving_utilization'],
            'age': fd['age'],
            'NumberOfTime30-59DaysPastDueNotWorse': fd['past_due_30_59'],
            'DebtRatio': fd['debt_ratio'],
            'MonthlyIncome': fd['monthly_income'],
            'NumberOfOpenCreditLinesAndLoans': fd['open_lines'],
            'NumberOfTimes90DaysLate': fd['past_due_90_plus'],
            'NumberRealEstateLoansOrLines': fd['real_estate_lines'],
            'NumberOfTime60-89DaysPastDueNotWorse': fd['past_due_60_89'],
            'NumberOfDependents': fd['dependents']
        }])
        
        with st.container(border=True):
            st.subheader("Prediction Results", anchor=False)
            with st.spinner("Analyzing profile patterns with ML Engine..."):
                prob = model.predict_proba(input_data)[0][1]
                
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Calculated Default Probability", value=f"{prob * 100:.1f}%")
                
            with res_col2:
                if prob < 0.1:
                    st.success("### Excellent Profile")
                    st.markdown("This applicant has a very low risk of defaulting within the next 2 years.")
                    progress_color = "#10b981" # Emerald gradient capable
                elif prob < 0.3:
                    st.warning("### Moderate Risk")
                    st.markdown("This applicant presents a moderate risk. Further review of credit history may be necessary.")
                    progress_color = "#f59e0b" # Amber
                else:
                    st.error("### High Risk")
                    st.markdown("This applicant presents a high risk of defaulting. Careful consideration advised.")
                    progress_color = "#ef4444" # Red
                    
                # Added a visual progress bar indicating risk level with animation
                st.markdown(f'''
                    <div style="width: 100%; background-color: #2D3748; border-radius: 999px; height: 16px; margin-top: 15px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);">
                      <div style="width: {min(prob * 100, 100)}%; background-color: {progress_color}; height: 100%; transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1); border-radius: 999px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #A0AEC0; margin-top: 8px; font-weight: 500;">
                        <span>Safe (0%)</span>
                        <span>High Risk (100%)</span>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Reset Assessment Form", use_container_width=False):
                    st.session_state.step = 1
                    st.session_state.analyze_trigger = False
                    st.rerun()

elif page == "Data Insights Dashboard":
    st.title("Data Insights & Exploration", anchor=False)
    st.markdown("<div class='info-box'>Explore a sample of the historical banking dataset used to train our AI model. Understand key demographic and financial trends.</div>", unsafe_allow_html=True)
    
    df = load_data()
    
    if df.empty:
        st.error("⚠️ Could not load the dataset for insights. Ensure `cs-training.csv` is in the directory.")
    else:
        # Overview Cards
        st.subheader("Dataset Snapshot", anchor=False)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sample Size Analyzed", f"{len(df):,}")
        m2.metric("Average Age", f"{int(df['age'].mean())} yrs")
        
        # Calculate overall default rate in the sample
        default_rate = df['SeriousDlqin2yrs'].mean() * 100
        m3.metric("Baseline Default Rate", f"{default_rate:.1f}%")
        m4.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Charts
        tab1, tab2, tab3 = st.tabs(["Age Demographics", "Income & Debt", "Data Table"])
        
        with tab1:
            st.markdown("#### Does age impact loan defaulting?", anchor=False)
            st.markdown("<p style='color: #A0AEC0;'>Generally, younger borrowers exhibit slightly higher default rates as they establish their financial profiles.</p>", unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                chart_age = alt.Chart(df).mark_bar(color='#3b82f6', cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    alt.X("age:Q", bin=alt.Bin(maxbins=20), title="Borrower Age"),
                    alt.Y('count()', title="Number of Borrowers"),
                    tooltip=['count()']
                ).properties(height=350).interactive()
                st.altair_chart(chart_age, use_container_width=True)
                
            with chart_col2:
                df['Age Group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '60+'])
                def_rate = df.groupby('Age Group', observed=False)['SeriousDlqin2yrs'].mean().reset_index()
                def_rate['SeriousDlqin2yrs_pct'] = def_rate['SeriousDlqin2yrs'] * 100
                
                chart_rate = alt.Chart(def_rate).mark_bar(color='#ef4444', cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    alt.X('Age Group:N', title="Age Group"),
                    alt.Y('SeriousDlqin2yrs_pct:Q', title="Default Rate (%)"),
                    tooltip=['Age Group', alt.Tooltip('SeriousDlqin2yrs_pct', format='.1f', title='Default Rate (%)')]
                ).properties(height=350)
                st.altair_chart(chart_rate, use_container_width=True)

        with tab2:
            st.markdown("#### How does income correlate with debt obligations?", anchor=False)
            st.markdown("<p style='color: #A0AEC0;'>Hover over the points in the scatter plot below. Outliers with high debt ratios represent stressed profiles.</p>", unsafe_allow_html=True)
            
            filtered_df = df[df['MonthlyIncome'] < 25000].dropna(subset=['MonthlyIncome', 'DebtRatio']).sample(min(1500, len(df)))
            chart_scatter = alt.Chart(filtered_df).mark_circle(size=70, opacity=0.5, color='#a78bfa').encode(
                x=alt.X('MonthlyIncome:Q', title="Monthly Income ($)"),
                y=alt.Y('DebtRatio:Q', title="Debt Ratio", scale=alt.Scale(domain=[0, 5], clamp=True)),
                tooltip=['age', 'MonthlyIncome', 'DebtRatio', 'SeriousDlqin2yrs']
            ).properties(height=400).interactive()
            st.altair_chart(chart_scatter, use_container_width=True)
            
        with tab3:
            st.markdown("#### Raw Sample Viewer", anchor=False)
            st.dataframe(df.head(100), use_container_width=True)

elif page == "System Architecture":
    st.title("System Architecture", anchor=False)
    st.markdown("<div class='info-box'>Building robust financial safety nets through machine learning technology.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### **The Problem Statement**
        Credit scoring is a critical process by which banks determine the risk associated with lending money. This application utilizes historical anonymized datasets and machine learning to predict the probability that a prospective borrower will experience financial distress in the next two years.
        
        ### **Engine & Model Details**
        The intelligence behind this web tier is a `HistGradientBoostingClassifier`, a gradient boosting machine optimized for speed and large tabular datasets. 
        - **Training Target**: `SeriousDlqin2yrs` (Binary classification of 90+ days past due or worse)
        - **Handling Missing Values**: Robust median imputation
        
        ### **Key Predictive Factors**
        1. **Delinquency History**: The sheer count of times a borrower has historically failed to pay within a 30, 60, or 90 day window is the strongest predictor of future default.
        2. **Utilization**: `Revolving Utilization of Unsecured Lines` (e.g. maxing out credit cards) heavily inflates risk signals.
        3. **Debt-to-Income**: Individuals operating with severe debt ratios are flagged securely.
        """, anchor=False)
        
    with col2:
        with st.container(border=True):
            st.markdown("#### Stack Profile")
            st.markdown("- **Frontend**: Streamlit 1.x")
            st.markdown("- **Theme**: Deep Space Black")
            st.markdown("- **Visualization**: Altair + Tooltips")
            st.markdown("- **Backend**: Python 3")
            st.markdown("- **AI Framework**: Scikit-Learn")
            st.markdown("- **Serializer**: Joblib")
