import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, date, time, timedelta, timezone
import requests
import io
import plotly.express as px
import plotly.graph_objects as go

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    # Basic text cleaning: remove non-alphanumeric characters, convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def remove_bias(text):
    # List of common bias-sensitive words (gender, nationality, age indicators)
    bias_words = [
        r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b', r'\bhers\b', 
        r'\bmr\b', r'\bmrs\b', r'\bms\b', r'\bmiss\b', r'\bman\b', r'\bwoman\b', 
        r'\bmen\b', r'\bwomen\b', r'\bamerican\b', r'\bindian\b', r'\basian\b', 
        r'\beuropean\b', r'\bafrican\b', r'\blatino\b', r'\bhispanic\b', 
        r'\byoung\b', r'\bold\b', r'\bage\b', r'\bdob\b', r'\bgender\b', r'\bnationality\b'
    ]
    for word in bias_words:
        text = re.sub(word, '', text, flags=re.IGNORECASE)
    # Return text with extra spaces removed
    return re.sub(r'\s+', ' ', text).strip()

def match_resumes(job_description, resumes_text):
    documents = [job_description] + resumes_text
    
    # Vectorize documents using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity between the job description (index 0) and all resumes
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

# Global Country and City Database Mapping
COUNTRY_DATA = {
    "USA": {
        "cities": ["New York", "San Francisco", "Los Angeles", "Chicago", "Seattle", "Austin", "Boston"],
        "companies": "e.g. Google, Microsoft, Amazon, Meta",
        "currency": "USD ($)",
        "tz_offset": -5,
        "tz_name": "EST/EDT"
    },
    "UK": {
        "cities": ["London", "Manchester", "Birmingham", "Edinburgh", "Glasgow", "Bristol"],
        "companies": "e.g. BP, HSBC, Unilever, Barclays",
        "currency": "GBP (£)",
        "tz_offset": 0,
        "tz_name": "GMT/BST"
    },
    "Canada": {
        "cities": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton"],
        "companies": "e.g. Shopify, RBC, Thomson Reuters",
        "currency": "CAD ($)",
        "tz_offset": -5,
        "tz_name": "EST/EDT"
    },
    "Australia": {
        "cities": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra"],
        "companies": "e.g. Atlassian, Commonwealth Bank, BHP",
        "currency": "AUD ($)",
        "tz_offset": 10,
        "tz_name": "AEST"
    },
    "UAE": {
        "cities": ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Fujairah"],
        "companies": "e.g. Emirates, Etisalat, DP World, Emaar",
        "currency": "AED",
        "tz_offset": 4,
        "tz_name": "GST"
    },
    "Saudi Arabia": {
        "cities": ["Riyadh", "Jeddah", "Mecca", "Medina", "Dammam"],
        "companies": "e.g. Saudi Aramco, SABIC, STC, Al Rajhi Bank",
        "currency": "SAR",
        "tz_offset": 3,
        "tz_name": "AST"
    },
    "Pakistan": {
        "cities": ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Faisalabad", "Rawalpindi"],
        "companies": "e.g. Systems Limited, TCS Pakistan, Habib Bank",
        "currency": "PKR",
        "tz_offset": 5,
        "tz_name": "PKT"
    },
    "India": {
        "cities": ["Mumbai", "Bengaluru", "New Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"],
        "companies": "e.g. Tata, Infosys, Reliance, Wipro",
        "currency": "INR (₹)",
        "tz_offset": 5.5,
        "tz_name": "IST"
    },
    "Germany": {
        "cities": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Cologne", "Stuttgart"],
        "companies": "e.g. SAP, Volkswagen, Siemens, Allianz",
        "currency": "EUR (€)",
        "tz_offset": 1,
        "tz_name": "CET/CEST"
    },
    "France": {
        "cities": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Nantes"],
        "companies": "e.g. L'Oréal, TotalEnergies, BNP Paribas, Sanofi",
        "currency": "EUR (€)",
        "tz_offset": 1,
        "tz_name": "CET/CEST"
    }
}

def main():
    st.set_page_config(page_title="AI Recruitment Automation", page_icon="📄", layout="wide")
    
    st.title("📄 AI Recruitment Automation Dashboard")
    st.markdown("Automate your recruitment pipeline: Generate Job Postings, Screen Resumes with Bias Reduction, and Schedule Interviews.")
    
    global_css = """
<style>
/* Main background */
.stApp {
    background-color: #F0F4F8 !important;
}
.main {
    background-color: #F0F4F8 !important;
}
/* Sidebar background */
[data-testid="stSidebar"] { background-color: #1E3A5F !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: white !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
    opacity: 1 !important;
}
/* All headings */
h1, h2, h3, h4, h5, h6 { color: #1E3A5F !important; }
/* All text */
p, span, label, div { color: #2C3E50; }
/* Accent color for standard buttons / accents */
.stButton>button { background-color: #2E86C1 !important; color: white !important; border: none !important; border-radius: 8px !important; }
.stButton>button:hover { background-color: #1E3A5F !important; }

/* Metric Cards */
.metric-container { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card { border-radius: 12px; padding: 20px; flex: 1; min-width: 150px; background-color: #1E3A5F; border: 2px solid #2E86C1; text-align: center; transition: transform 0.2s ease, box-shadow 0.2s ease; }
.metric-card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(46, 134, 193, 0.3); }
.metric-icon { font-size: 28px; margin-bottom: 10px; display: block; }
.metric-value { font-size: 32px; font-weight: 700; color: white !important; margin: 0; line-height: 1.2; }
.metric-label { font-size: 14px; color: white !important; font-weight: 600; margin: 5px 0 0 0; }

/* Tabs Styling */
.stTabs [data-baseweb="tab"] {
    background-color: #E8F0FE !important;
    color: #1E3A5F !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #2E86C1 !important;
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #AED6F1 !important;
    color: #1E3A5F !important;
}

/* Dropdowns and Selectboxes */
div[data-baseweb="select"] * {
    color: #1E3A5F !important;
    background-color: #FFFFFF !important;
}
div[data-baseweb="popover"] * {
    color: #1E3A5F !important;
    background-color: #FFFFFF !important;
}

/* Input Fields */
.stTextInput input, .stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #1E3A5F !important;
}

/* Sidebar File Uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: #2E5F8A !important;
    border: 2px dashed #5DADE2 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #5DADE2 !important;
    color: #FFFFFF !important;
    border: none !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #AED6F1 !important;
}

/* Sidebar Dropdowns */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: transparent !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #FFFFFF !important;
    background-color: transparent !important;
}
</style>
"""
    st.markdown(global_css, unsafe_allow_html=True)
    

    
    st.sidebar.header("🌍 Worldwide Localization")
    
    # Step 1: Country Selection
    countries = list(COUNTRY_DATA.keys())
    selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index("Pakistan"))
    
    # Step 2: City Selection based on Country
    cd = COUNTRY_DATA[selected_country]
    cities = list(cd["cities"])
    selected_city = st.sidebar.selectbox("Select Major City", cities, index=cities.index("Karachi"))
    
    # --- Global Dashboard KPIs ---
    st.subheader("📊 Recruitment Overview")
    
    total_scanned = 0
    top_score = "0%"
    avg_score = "0%"
    
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        df = st.session_state['results_df']
        total_scanned = len(df)
        top_score = f"{df.iloc[0]['Match Score (%)']}%"
        avg_score = f"{round(df['Match Score (%)'].mean(), 2)}%"
        
    kpi_html = f'''
<div class="metric-container">
<div class="metric-card tile-1">
<span class="metric-icon">📄</span>
<p class="metric-value">{total_scanned}</p>
<p class="metric-label">Total Resumes Scanned</p>
</div>
<div class="metric-card tile-2">
<span class="metric-icon">🏆</span>
<p class="metric-value">{top_score}</p>
<p class="metric-label">Top Match Score</p>
</div>
<div class="metric-card tile-3">
<span class="metric-icon">📈</span>
<p class="metric-value">{avg_score}</p>
<p class="metric-label">Avg Match Score</p>
</div>
</div>
'''
    st.markdown(kpi_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Analytics Section ---
    st.markdown("---")
    st.subheader("📈 Recruitment Analytics")
    st.markdown("Track pipeline health, candidate qualification, and skill distribution.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 🎯 Qualification Status")
        st.markdown("<span style='color:gray; font-size:14px;'>Overview of candidates who passed the 50% match threshold.</span>", unsafe_allow_html=True)
        if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
            df = st.session_state['results_df']
            qualified = len(df[df['Match Score (%)'] > 50])
            unqualified = len(df) - qualified
            
            fig_pie = px.pie(
                names=['Qualified (>50%)', 'Unqualified (<=50%)'], 
                values=[qualified, unqualified],
                hole=0.4,
                color_discrete_sequence=['#27AE60', '#5DADE2']
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Upload and scan resumes to view the Qualified vs Unqualified chart.")
            
    with chart_col2:
        st.markdown("#### 🛠️ Skills Distribution")
        st.markdown("<span style='color:gray; font-size:14px;'>Most frequent technical skills identified across all scanned resumes.</span>", unsafe_allow_html=True)
        if 'skill_counts' in st.session_state:
            skills = st.session_state['skill_counts']
            active_skills = {k.title(): v for k, v in skills.items() if v > 0}
            if active_skills:
                df_skills = pd.DataFrame(list(active_skills.items()), columns=['Skill', 'Count']).sort_values(by='Count', ascending=True)
                fig_bar = px.bar(
                    df_skills, 
                    x='Count', 
                    y='Skill', 
                    orientation='h',
                    color='Count',
                    color_continuous_scale=['#AED6F1', '#5DADE2', '#2E86C1', '#1E3A5F']
                )
                fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Count", yaxis_title="")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No common skills found in scanned resumes.")
        else:
            st.info("Upload and scan resumes to view the Top Skills Distribution.")
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### 🚦 Hiring Funnel")
    st.markdown("<span style='color:gray; font-size:14px;'>End-to-end conversion tracking from sourcing leads to scheduling interviews.</span>", unsafe_allow_html=True)
    
    profiles_found = len(st.session_state.get('linkedin_profiles', []))
    cvs_uploaded = len(st.session_state.get('results_df', []))
    
    qualified_cands = 0
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        qualified_cands = len(st.session_state['results_df'][st.session_state['results_df']['Match Score (%)'] > 50])
        
    interviews_scheduled = len(st.session_state.get('schedule_df', []))
    
    if profiles_found > 0 or cvs_uploaded > 0 or interviews_scheduled > 0:
        fig_funnel = go.Figure(go.Funnel(
            y=["Sourced Profiles", "CVs Screened", "Qualified Candidates", "Interviews Scheduled"],
            x=[profiles_found, cvs_uploaded, qualified_cands, interviews_scheduled],
            textinfo="value+percent initial",
            marker={"color": ["#1E3A5F", "#2E86C1", "#5DADE2", "#AED6F1"]}
        ))
        fig_funnel.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_funnel, use_container_width=True)
    else:
        st.info("Interact with the dashboard (Search Profiles, Upload CVs, Schedule) to populate the Hiring Funnel.")
        
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Job Posting Generator", "📊 Resume Screening", "📅 Interview Scheduling", "🔍 LinkedIn Profile Search", "💰 Salary Insights"])
    
    # --- Feature 1: Automated Job Posting ---
    with tab1:
        st.header("Automated Job Posting")
        st.markdown("Generate a professional job posting dynamically tailored to your region.")
        
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g. Senior Data Scientist")
            # Auto-suggest companies based on country
            company_name = st.text_input("Company Name", placeholder=cd["companies"])
            # Auto-filled location based on sidebar selection
            location = st.text_input("Location", value=f"{selected_city}, {selected_country}")
        with col2:
            req_skills = st.text_input("Required Skills", placeholder="e.g. Python, SQL, Machine Learning")
            # Auto-suggest currency based on country
            salary_range = st.text_input("Salary Range", placeholder=f"e.g. 80,000 - 120,000 {cd['currency']}")
            
        if st.button("Generate Job Posting"):
            if job_title and company_name:
                posting = f"""### {job_title} at {company_name}

**Location:** {location}

**About the Role:**
We are looking for a highly motivated {job_title} to join our dynamic team at {company_name}. 
You will be responsible for driving impactful projects and working closely with cross-functional teams.

**Required Skills:**
- {req_skills}

**Compensation:**
- {salary_range}

**How to Apply:**
Please submit your resume and cover letter. We are an equal opportunity employer and value diversity."""
                st.subheader("Generated Posting:")
                st.code(posting, language="markdown")
            else:
                st.warning("Please enter at least the Job Title and Company Name.")

    # --- Feature 2: Bias Reduction & Resume Screening ---
    with tab2:
        st.header("Resume Screening (Bias Reduced)")
        
        st.sidebar.header("📄 Resume Screening Inputs")
        job_description = st.sidebar.text_area("Job Description", height=300, placeholder="Paste the job description here...")
        uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Scan and Match Resumes"):
            if not job_description:
                st.warning("Please enter a job description in the sidebar.")
            elif not uploaded_files:
                st.warning("Please upload at least one candidate resume.")
            else:
                with st.spinner("Analyzing and scoring candidates..."):
                    resumes_text = []
                    file_names = []
                    
                    for file in uploaded_files:
                        pdf_text = extract_text_from_pdf(file)
                        
                        # Apply Bias Reduction
                        text_unbiased = remove_bias(pdf_text)
                        cleaned_text = clean_text(text_unbiased)
                        
                        resumes_text.append(cleaned_text)
                        file_names.append(file.name)
                    
                    cleaned_jd = clean_text(job_description)
                    scores = match_resumes(cleaned_jd, resumes_text)
                    scores_percent = [round(score * 100, 2) for score in scores]
                    
                    # Extract Skills Distribution
                    tech_skills = ["python", "sql", "machine learning", "java", "aws", "azure", "gcp", "react", "node", "excel", "data analysis", "agile", "communication", "leadership"]
                    skill_counts = {skill: 0 for skill in tech_skills}
                    for text in resumes_text:
                        for skill in tech_skills:
                            if re.search(rf'\b{re.escape(skill)}\b', text):
                                skill_counts[skill] += 1
                                
                    st.session_state['skill_counts'] = skill_counts
                    
                    results_df = pd.DataFrame({
                        "Candidate Name / File": file_names,
                        "Status": ["Bias Removed ✅"] * len(file_names),
                        "Match Score (%)": scores_percent
                    })
                    
                    results_df = results_df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
                    st.session_state['results_df'] = results_df
                    st.rerun()

        if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
            df = st.session_state['results_df']
            st.subheader("🏆 Candidate Rankings")
            st.dataframe(df, use_container_width=True)
            
            top_candidate = df.iloc[0]['Candidate Name / File']
            top_score = df.iloc[0]['Match Score (%)']
            st.success(f"**{top_candidate}** is the best match with a relevance score of **{top_score}%**.")

    # --- Feature 3: Interview Scheduling ---
    with tab3:
        st.header("Interview Scheduling")
        if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
            results_df = st.session_state['results_df']
            
            st.markdown(f"Schedule interviews in **{selected_country} ({cd['tz_name']})** for your top candidates.")
            
            num_candidates = st.number_input("Select Top N Candidates to Schedule", min_value=1, max_value=len(results_df), value=min(3, len(results_df)))
            top_candidates = results_df.head(num_candidates)['Candidate Name / File'].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                # Retaining standard date format visual UI
                try:
                    interview_date = st.date_input("Start Date for Interviews", date.today(), format="DD/MM/YYYY")
                except TypeError:
                    interview_date = st.date_input("Start Date for Interviews", date.today())
            with col2:
                interview_time = st.time_input("Start Time (First Interview)", time(9, 0))
                
            interview_type = st.selectbox("Interview Type", ["HR Screen", "Technical Interview", "Cultural Fit", "Final Round"])
            
            if st.button("Generate Schedule"):
                schedule_data = []
                
                # Setup Dynamic Timezone based on country offset
                offset_val = cd["tz_offset"]
                hours = int(offset_val)
                minutes = int((offset_val - hours) * 60)
                dynamic_tz = timezone(timedelta(hours=hours, minutes=minutes), name=cd["tz_name"])
                current_time = datetime.combine(interview_date, interview_time).replace(tzinfo=dynamic_tz)
                
                for cand in top_candidates:
                    schedule_data.append({
                        "Candidate Name": cand,
                        "Assigned Date": current_time.strftime("%d/%m/%Y"), 
                        "Assigned Time": current_time.strftime(f"%I:%M %p ({cd['tz_name']})"), 
                        "Interview Type": interview_type
                    })
                    # Schedule candidates 1 hour apart consecutively
                    current_time += timedelta(hours=1)
                    
                schedule_df = pd.DataFrame(schedule_data)
                st.session_state['schedule_df'] = schedule_df
                
                st.subheader("📅 Generated Interview Schedule")
                st.table(schedule_df)
                st.success("Interview schedule generated successfully!")
                
        else:
            st.info("Please scan and match resumes in the 'Resume Screening' tab first to generate a schedule.")

    # --- Feature 4: LinkedIn Profile Search ---
    with tab4:
        st.header("🔍 HR Leads Dashboard")
        st.markdown("Search and manage LinkedIn professionals matching your criteria.")
        
        search_col, btn_col = st.columns([4, 1])
        with search_col:
            search_keyword = st.text_input("Search Keyword", placeholder="e.g. Python Developer", label_visibility="collapsed")
        with btn_col:
            search_btn = st.button("Search Profiles", use_container_width=True)
            
        if search_btn:
            if not search_keyword:
                st.warning("Please enter a keyword to search.")
            else:
                st.session_state['last_search_keyword'] = search_keyword
                with st.spinner("Fetching profiles from LinkedIn via RapidAPI..."):
                    import http.client
                    import json
                    import urllib.parse
                    
                    try:
                        conn = http.client.HTTPSConnection("linkedin-profile-search-api-by-name-job-title-company.p.rapidapi.com")
                        headers = {
                            'x-rapidapi-key': "11de715bc1msh4cd9d1e5a295836p186943jsn158b752bf364",
                            'x-rapidapi-host': "linkedin-profile-search-api-by-name-job-title-company.p.rapidapi.com",
                            'Content-Type': "application/json"
                        }
                        
                        full_search_keyword = f"{search_keyword} {selected_city} {selected_country}"
                        encoded_keyword = urllib.parse.quote(full_search_keyword)
                        conn.request("GET", f"/sync?k={encoded_keyword}&linkedin_type=individual&depth=3", headers=headers)
                        
                        res = conn.getresponse()
                        raw_data = res.read().decode("utf-8")
                        data = json.loads(raw_data)
                        
                        st.session_state['raw_api_data'] = data
                        
                        if isinstance(data, dict):
                            profiles = data.get("data", data.get("items", data.get("people", data.get("results", []))))
                        elif isinstance(data, list):
                            profiles = data
                        else:
                            profiles = []
                            
                        if not profiles:
                            st.info("No profiles found or API rate limit exceeded.")
                        else:
                            extracted_profiles = []
                            for p in profiles:
                                raw_title = p.get("title", "")
                                if " - " in raw_title:
                                    parts = raw_title.split(" - ", 1)
                                    name = parts[0].strip()
                                    job_title = parts[1].strip()
                                else:
                                    name = raw_title if raw_title else "N/A"
                                    job_title = "N/A"
                                    
                                summary = p.get("description", "N/A")
                                
                                skills = "N/A"
                                education = "N/A"
                                experience = "N/A"
                                
                                if summary != "N/A":
                                    # Extract Skills
                                    tech_skills = ["Python", "SQL", "Machine Learning", "Power BI", "Tableau", "NLP", "Deep Learning", "Java", "C++", "AWS", "Azure", "GCP", "React", "Angular", "Node.js", "Excel", "Data Analysis", "TensorFlow", "PyTorch"]
                                    found_skills = []
                                    for skill in tech_skills:
                                        if re.search(rf'\b{re.escape(skill)}\b', summary, re.IGNORECASE):
                                            found_skills.append(skill)
                                    if found_skills:
                                        skills = ", ".join(found_skills)
                                        
                                    # Extract Education
                                    edu_match = re.search(r'Education:\s*([^\n]*)', summary, re.IGNORECASE)
                                    if edu_match:
                                        education = edu_match.group(1).strip()
                                    else:
                                        edu_keywords = ["University", "College", "Institute"]
                                        for kw in edu_keywords:
                                            kw_match = re.search(rf'([^,.\n]*{kw}[^,.\n]*)', summary, re.IGNORECASE)
                                            if kw_match:
                                                education = kw_match.group(1).strip()
                                                break
                                                
                                    # Extract Experience
                                    exp_match = re.search(r'Experience:\s*([^\n]*)', summary, re.IGNORECASE)
                                    if exp_match:
                                        experience = exp_match.group(1).strip()
                                    else:
                                        at_match = re.search(r'\bat\s+([A-Z][a-zA-Z0-9\s&]+?)(?=[.,\n]|$)', summary)
                                        if at_match and len(at_match.group(1).split()) <= 4:
                                            experience = at_match.group(1).strip()
                                                
                                linkedin_url = p.get("url", "N/A")
                                
                                extracted_profiles.append({
                                    "Name": name,
                                    "Job Title": job_title,
                                    "Skills": skills,
                                    "Education": education,
                                    "Experience": experience,
                                    "Summary": summary,
                                    "LinkedIn URL": linkedin_url
                                })
                                
                            df_profiles = pd.DataFrame(extracted_profiles)
                            df_profiles.index = df_profiles.index + 1
                            st.session_state['linkedin_profiles'] = df_profiles
                            
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
        
        if 'raw_api_data' in st.session_state:
            with st.expander("🛠️ Debug: Show Raw API Response"):
                st.json(st.session_state['raw_api_data'])
                
        if 'linkedin_profiles' in st.session_state and not st.session_state['linkedin_profiles'].empty:
            df_profiles = st.session_state['linkedin_profiles']
            
            # --- Top Stats Bar ---
            st.markdown("### 📊 Leads Dashboard Overview")
            
            industry = st.session_state.get('last_search_keyword', 'General')
            if " " in industry:
                industry = industry.split()[0]
                
            conversion_est = min(100.0, len(df_profiles) * 3.5)
            
            metrics_html = f'''
<div class="metric-container">
<div class="metric-card tile-1">
<span class="metric-icon">👥</span>
<p class="metric-value">{len(df_profiles)}</p>
<p class="metric-label">Total Leads Found</p>
</div>
<div class="metric-card tile-2">
<span class="metric-icon">📍</span>
<p class="metric-value">{selected_city}</p>
<p class="metric-label">Target Location</p>
</div>
<div class="metric-card tile-3">
<span class="metric-icon">🎯</span>
<p class="metric-value">{industry.capitalize()}</p>
<p class="metric-label">Industry Focus</p>
</div>
<div class="metric-card tile-4">
<span class="metric-icon">📈</span>
<p class="metric-value">{conversion_est:.1f}%</p>
<p class="metric-label">Conversion Est.</p>
</div>
</div>
'''
            st.markdown(metrics_html, unsafe_allow_html=True)

            st.markdown("---")

            # --- Filters and Search ---
            filt_col, search_col2 = st.columns([2, 1])
            with filt_col:
                filter_option = st.radio("Filter by Role Level:", ["All", "Managers", "Directors", "Engineers"], horizontal=True)
            with search_col2:
                local_search = st.text_input("🔍 Search results...", placeholder="Name or Title")

            # Apply filters
            filtered_df = df_profiles.copy()
            if filter_option == "Managers":
                filtered_df = filtered_df[filtered_df['Job Title'].str.contains('Manager|Head|Lead', case=False, na=False)]
            elif filter_option == "Directors":
                filtered_df = filtered_df[filtered_df['Job Title'].str.contains('Director|VP|President', case=False, na=False)]
            elif filter_option == "Engineers":
                filtered_df = filtered_df[filtered_df['Job Title'].str.contains('Engineer|Developer|Programmer', case=False, na=False)]
                
            if local_search:
                filtered_df = filtered_df[
                    filtered_df['Name'].str.contains(local_search, case=False, na=False) | 
                    filtered_df['Job Title'].str.contains(local_search, case=False, na=False)
                ]

            # --- Export CSV Button ---
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            
            col_exp, _ = st.columns([1, 4])
            with col_exp:
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name="hr_leads_database.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- HR Professionals Database Table ---
            st.markdown("### 🧑‍💼 HR Professionals Database")
            
            html_content = '''
<style>
.lead-table {
width: 100%;
border-collapse: collapse;
font-family: sans-serif;
margin-top: 10px;
}
.lead-table th {
background-color: #2E86C1;
color: white !important;
padding: 12px;
text-align: left;
border-bottom: 2px solid var(--border-color);
font-weight: 600;
}
.lead-table td {
padding: 12px;
border-bottom: 1px solid var(--border-color);
color: var(--text-color);
vertical-align: middle;
}
.avatar {
width: 40px;
height: 40px;
min-width: 40px;
border-radius: 50%;
background-color: #2E86C1;
color: white !important;
display: inline-flex;
align-items: center;
justify-content: center;
font-weight: bold;
font-size: 16px;
margin-right: 15px;
}
.name-cell {
display: flex;
align-items: center;
}
.status-badge {
background-color: #27AE60;
color: white !important;
padding: 4px 10px;
border-radius: 12px;
font-size: 12px;
font-weight: bold;
display: inline-block;
}
.view-btn {
background-color: #2E86C1;
color: white !important;
padding: 8px 16px;
text-decoration: none;
border-radius: 4px;
font-size: 14px;
font-weight: 500;
display: inline-block;
text-align: center;
}
.view-btn:hover {
background-color: #1E3A5F;
}
</style>
<table class="lead-table">
<thead>
<tr>
<th>Candidate Name</th>
<th>Headline / Job Title</th>
<th>Status</th>
<th>Action</th>
</tr>
</thead>
<tbody>
'''

            if filtered_df.empty:
                html_content += '''
<tr>
<td colspan="4" style="text-align: center; padding: 20px; color: gray;">
No profiles match the selected filters.
</td>
</tr>
'''
            else:
                for _, row in filtered_df.iterrows():
                    name = str(row['Name']).replace("<", "&lt;").replace(">", "&gt;")
                    initials = "".join([n[0] for n in name.split() if n.isalnum()])[:2].upper() if name and name != "N/A" else "??"
                    title = str(row['Job Title']).replace("<", "&lt;").replace(">", "&gt;")
                    url = str(row['LinkedIn URL']).replace('"', '&quot;')
                    
                    html_content += f'''
<tr>
<td>
<div class="name-cell">
<div class="avatar">{initials}</div>
<strong>{name}</strong>
</div>
</td>
<td>{title}</td>
<td><span class="status-badge">● Active</span></td>
<td><a href="{url}" target="_blank" class="view-btn">View Profile</a></td>
</tr>
'''
                    
            html_content += "</tbody>\n</table>"
            
            st.markdown(html_content, unsafe_allow_html=True)

    # --- Feature 5: Salary Insights ---
    with tab5:
        st.header("💰 Salary Insights (Pakistan Market)")
        st.markdown("Explore compensation trends and market insights for top tech roles in Pakistan.")
        
        salary_data = {
            "Data Scientist": {"Entry": [150000, 250000], "Mid": [250000, 450000], "Senior": [450000, 800000]},
            "Software Engineer": {"Entry": [100000, 200000], "Mid": [200000, 400000], "Senior": [400000, 700000]},
            "Business Analyst": {"Entry": [80000, 150000], "Mid": [150000, 300000], "Senior": [300000, 500000]},
            "ML Engineer": {"Entry": [180000, 300000], "Mid": [300000, 550000], "Senior": [550000, 900000]},
            "Data Analyst": {"Entry": [80000, 150000], "Mid": [150000, 250000], "Senior": [250000, 400000]},
            "HR Manager": {"Entry": [120000, 200000], "Mid": [200000, 350000], "Senior": [350000, 600000]},
            "Product Manager": {"Entry": [150000, 250000], "Mid": [250000, 500000], "Senior": [500000, 900000]},
            "DevOps Engineer": {"Entry": [150000, 250000], "Mid": [250000, 450000], "Senior": [450000, 800000]},
            "Cybersecurity Analyst": {"Entry": [120000, 220000], "Mid": [220000, 400000], "Senior": [400000, 750000]},
            "UI/UX Designer": {"Entry": [80000, 150000], "Mid": [150000, 280000], "Senior": [280000, 450000]},
            "Financial Analyst": {"Entry": [80000, 150000], "Mid": [150000, 300000], "Senior": [300000, 500000]},
            "Marketing Manager": {"Entry": [100000, 180000], "Mid": [180000, 350000], "Senior": [350000, 600000]},
            "Project Manager": {"Entry": [120000, 200000], "Mid": [200000, 380000], "Senior": [380000, 650000]},
            "Cloud Architect": {"Entry": [200000, 350000], "Mid": [350000, 600000], "Senior": [600000, 1000000]},
            "Full Stack Developer": {"Entry": [120000, 220000], "Mid": [220000, 450000], "Senior": [450000, 800000]}
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Salary Check")
            sel_title = st.selectbox("Select Job Title", list(salary_data.keys()))
            sel_exp = st.select_slider("Experience Level", options=["Entry", "Mid", "Senior"])
            
            s_min = salary_data[sel_title][sel_exp][0]
            s_max = salary_data[sel_title][sel_exp][1]
            s_avg = (s_min + s_max) // 2
            
            st.metric("Expected Average Salary", f"PKR {s_avg:,}")
            st.caption(f"Range: PKR {s_min:,} - PKR {s_max:,}")
            
            st.markdown("---")
            st.subheader("🏢 Market Insights")
            st.markdown("**Top Hiring Companies in Pakistan:**")
            st.markdown("- Systems Limited\n- Motive (KeepTruckin)\n- Afiniti\n- Arbisoft\n- 10Pearls\n- Careem\n- Contour Software")
            
        with col2:
            st.subheader("Market Salary Comparison")
            
            plot_data = []
            for title, levels in salary_data.items():
                for lvl, (vmin, vmax) in levels.items():
                    vavg = (vmin + vmax) // 2
                    plot_data.append({"Job Title": title, "Level": lvl, "Min Salary": vmin, "Avg Salary": vavg, "Max Salary": vmax})
            
            df_salary = pd.DataFrame(plot_data)
            
            fig_range = go.Figure()
            colors = {"Entry": "#5DADE2", "Mid": "#2E86C1", "Senior": "#1E3A5F"}
            
            for lvl in ["Entry", "Mid", "Senior"]:
                df_lvl = df_salary[df_salary["Level"] == lvl]
                fig_range.add_trace(go.Bar(
                    name=lvl,
                    x=df_lvl["Job Title"],
                    y=df_lvl["Max Salary"] - df_lvl["Min Salary"],
                    base=df_lvl["Min Salary"],
                    marker_color=colors[lvl],
                    text=df_lvl["Avg Salary"].apply(lambda x: f"{int(x/1000)}k"),
                    textposition="inside",
                    hovertemplate="<b>%{x}</b><br>Level: " + lvl + "<br>Min: PKR %{base:,}<br>Max: PKR %{customdata:,}<extra></extra>",
                    customdata=df_lvl["Max Salary"]
                ))
                
            fig_range.update_layout(
                title="Salary Ranges (Min to Max) in PKR",
                barmode="group",
                yaxis_title="Monthly Salary (PKR)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
            )
            st.plotly_chart(fig_range, use_container_width=True)

if __name__ == "__main__":
    main()
