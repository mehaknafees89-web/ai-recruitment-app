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
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def remove_bias(text):
    bias_words = [
        r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b', r'\bhers\b', 
        r'\bmr\b', r'\bmrs\b', r'\bms\b', r'\bmiss\b', r'\bman\b', r'\bwoman\b', 
        r'\bmen\b', r'\bwomen\b', r'\bamerican\b', r'\bindian\b', r'\basian\b', 
        r'\beuropean\b', r'\bafrican\b', r'\blatino\b', r'\bhispanic\b', 
        r'\byoung\b', r'\bold\b', r'\bage\b', r'\bdob\b', r'\bgender\b', r'\bnationality\b'
    ]
    for word in bias_words:
        text = re.sub(word, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def match_resumes(job_description, resumes_text):
    documents = [job_description] + resumes_text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

COUNTRY_DATA = {
    "USA": {"cities": ["New York", "San Francisco", "Los Angeles", "Chicago", "Seattle", "Austin", "Boston"], "companies": "e.g. Google, Microsoft, Amazon, Meta", "currency": "USD ($)", "tz_offset": -5, "tz_name": "EST/EDT"},
    "UK": {"cities": ["London", "Manchester", "Birmingham", "Edinburgh", "Glasgow", "Bristol"], "companies": "e.g. BP, HSBC, Unilever, Barclays", "currency": "GBP (£)", "tz_offset": 0, "tz_name": "GMT/BST"},
    "Canada": {"cities": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton"], "companies": "e.g. Shopify, RBC, Thomson Reuters", "currency": "CAD ($)", "tz_offset": -5, "tz_name": "EST/EDT"},
    "Australia": {"cities": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra"], "companies": "e.g. Atlassian, Commonwealth Bank, BHP", "currency": "AUD ($)", "tz_offset": 10, "tz_name": "AEST"},
    "UAE": {"cities": ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Fujairah"], "companies": "e.g. Emirates, Etisalat, DP World, Emaar", "currency": "AED", "tz_offset": 4, "tz_name": "GST"},
    "Saudi Arabia": {"cities": ["Riyadh", "Jeddah", "Mecca", "Medina", "Dammam"], "companies": "e.g. Saudi Aramco, SABIC, STC, Al Rajhi Bank", "currency": "SAR", "tz_offset": 3, "tz_name": "AST"},
    "Pakistan": {"cities": ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Faisalabad", "Rawalpindi"], "companies": "e.g. Systems Limited, TCS Pakistan, Habib Bank", "currency": "PKR", "tz_offset": 5, "tz_name": "PKT"},
    "India": {"cities": ["Mumbai", "Bengaluru", "New Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"], "companies": "e.g. Tata, Infosys, Reliance, Wipro", "currency": "INR (₹)", "tz_offset": 5.5, "tz_name": "IST"},
    "Germany": {"cities": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Cologne", "Stuttgart"], "companies": "e.g. SAP, Volkswagen, Siemens, Allianz", "currency": "EUR (€)", "tz_offset": 1, "tz_name": "CET/CEST"},
    "France": {"cities": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Nantes"], "companies": "e.g. L'Oréal, TotalEnergies, BNP Paribas, Sanofi", "currency": "EUR (€)", "tz_offset": 1, "tz_name": "CET/CEST"}
}

def main():
    st.set_page_config(page_title="AI Recruitment Automation", page_icon="📄", layout="wide")
    
    st.title("📄 AI Recruitment Automation Dashboard")
    st.markdown("Automate your recruitment pipeline: Screen Resumes, Schedule Interviews, and Generate Job Postings.")
    
    global_css = """
<style>
/* Main background */
.stApp { background-color: #F0F4F8 !important; }
.main { background-color: #F0F4F8 !important; }
/* Sidebar background */
[data-testid="stSidebar"] { background-color: #1E3A5F !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: white !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span { color: #FFFFFF !important; opacity: 1 !important; }
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
.stTabs [data-baseweb="tab"] { background-color: #E8F0FE !important; color: #1E3A5F !important; font-weight: 600 !important; border-radius: 8px 8px 0 0 !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #2E86C1 !important; color: #FFFFFF !important; }
.stTabs [data-baseweb="tab"]:hover { background-color: #AED6F1 !important; color: #1E3A5F !important; }
/* Dropdowns and Selectboxes */
div[data-baseweb="select"] * { color: #1E3A5F !important; background-color: #FFFFFF !important; }
div[data-baseweb="popover"] * { color: #1E3A5F !important; background-color: #FFFFFF !important; }
/* Input Fields */
.stTextInput input, .stTextArea textarea { background-color: #FFFFFF !important; color: #1E3A5F !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #95A5A6 !important; opacity: 1 !important; }
/* Sidebar File Uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] { background-color: #2E5F8A !important; border: 2px dashed #5DADE2 !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button { background-color: #5DADE2 !important; color: #FFFFFF !important; border: none !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color: #AED6F1 !important; }
/* Sidebar Dropdowns */
section[data-testid="stSidebar"] div[data-baseweb="select"] { background-color: transparent !important; border: 1px solid rgba(255,255,255,0.3) !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #FFFFFF !important; background-color: transparent !important; }
/* Sidebar Text Area */
section[data-testid="stSidebar"] .stTextArea textarea { background-color: #FFFFFF !important; color: #1E3A5F !important; border: 1px solid #5DADE2 !important; border-radius: 6px !important; pointer-events: auto !important; cursor: text !important; }
/* Sidebar Collapse Button */
button[data-testid="collapsedControl"] { background-color: #1E3A5F !important; border-radius: 50% !important; width: 40px !important; height: 40px !important; }
button[data-testid="collapsedControl"]::after { content: "☰" !important; color: white !important; font-size: 20px !important; }
button[data-testid="collapsedControl"] svg { display: none !important; }
</style>
"""
    st.markdown(global_css, unsafe_allow_html=True)
    
    st.sidebar.header("🌍 Worldwide Localization")
    
    countries = list(COUNTRY_DATA.keys())
    selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index("Pakistan"))
    
    cd = COUNTRY_DATA[selected_country]
    cities = list(cd["cities"])
    selected_city = st.sidebar.selectbox("Select Major City", cities, index=0)
    
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
    
    # 1. New Tab Order
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resume Screening", "📅 Interview Scheduling", "🔍 LinkedIn Profile Search", "📝 Job Posting Generator"])
    
    # --- Feature 1: Resume Screening ---
    with tab1:
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
                        text_unbiased = remove_bias(pdf_text)
                        cleaned_text = clean_text(text_unbiased)
                        
                        resumes_text.append(cleaned_text)
                        file_names.append(file.name)
                    
                    cleaned_jd = clean_text(job_description)
                    scores = match_resumes(cleaned_jd, resumes_text)
                    scores_percent = [round(score * 100, 2) for score in scores]
                    
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

    # --- Feature 2: Interview Scheduling ---
    with tab2:
        st.header("Interview Scheduling")
        if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
            results_df = st.session_state['results_df']
            
            st.markdown(f"Schedule interviews in **{selected_country} ({cd['tz_name']})** for your top candidates.")
            
            num_candidates = st.number_input("Select Top N Candidates to Schedule", min_value=1, max_value=len(results_df), value=min(3, len(results_df)))
            top_candidates = results_df.head(num_candidates)['Candidate Name / File'].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                try:
                    interview_date = st.date_input("Start Date for Interviews", date.today(), format="DD/MM/YYYY")
                except TypeError:
                    interview_date = st.date_input("Start Date for Interviews", date.today())
            with col2:
                interview_time = st.time_input("Start Time (First Interview)", time(9, 0))
                
            interview_type = st.selectbox("Interview Type", ["HR Screen", "Technical Interview", "Cultural Fit", "Final Round"])
            
            if st.button("Generate Schedule"):
                schedule_data = []
                
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
                    current_time += timedelta(hours=1)
                    
                schedule_df = pd.DataFrame(schedule_data)
                st.session_state['schedule_df'] = schedule_df
                
                st.subheader("📅 Generated Interview Schedule")
                st.table(schedule_df)
                st.success("Interview schedule generated successfully!")
                
        else:
            st.info("Please scan and match resumes in the 'Resume Screening' tab first to generate a schedule.")

    # --- Feature 3: LinkedIn Profile Search ---
    with tab3:
        st.markdown("# 🔍 LinkedIn Profile Search")
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
                        
                        # Fix 3: Proper error handling for empty API response
                        if res.status != 200:
                            st.error(f"API Error: Status {res.status}")
                        else:
                            raw_data = res.read().decode("utf-8")
                            
                            if not raw_data or not raw_data.strip():
                                st.info("No profiles found, please try different keyword")
                            else:
                                try:
                                    data = json.loads(raw_data)
                                    st.session_state['raw_api_data'] = data
                                    
                                    if isinstance(data, dict):
                                        profiles = data.get("data", data.get("items", data.get("people", data.get("results", []))))
                                    elif isinstance(data, list):
                                        profiles = data
                                    else:
                                        profiles = []
                                        
                                    if not profiles:
                                        st.info("No profiles found, please try different keyword")
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
                                                tech_skills = ["Python", "SQL", "Machine Learning", "Power BI", "Tableau", "NLP", "Deep Learning", "Java", "C++", "AWS", "Azure", "GCP", "React", "Angular", "Node.js", "Excel", "Data Analysis", "TensorFlow", "PyTorch"]
                                                found_skills = []
                                                for skill in tech_skills:
                                                    if re.search(rf'\b{re.escape(skill)}\b', summary, re.IGNORECASE):
                                                        found_skills.append(skill)
                                                if found_skills:
                                                    skills = ", ".join(found_skills)
                                                    
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
                                except json.JSONDecodeError:
                                    st.info("No profiles found, please try different keyword")
                                    
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
        
        if 'raw_api_data' in st.session_state:
            with st.expander("🛠️ Debug: Show Raw API Response"):
                st.json(st.session_state['raw_api_data'])
                
        if 'linkedin_profiles' in st.session_state and not st.session_state['linkedin_profiles'].empty:
            df_profiles = st.session_state['linkedin_profiles']
            
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

            filt_col, search_col2 = st.columns([2, 1])
            with filt_col:
                filter_option = st.radio("Filter by Role Level:", ["All", "Managers", "Directors", "Engineers"], horizontal=True)
            with search_col2:
                local_search = st.text_input("🔍 Search results...", placeholder="Name or Title")

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
            st.markdown("### 🧑‍💼 HR Professionals Database")
            
            html_content = '''
<style>
.lead-table { width: 100%; border-collapse: collapse; font-family: sans-serif; margin-top: 10px; }
.lead-table th { background-color: #2E86C1; color: white !important; padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color); font-weight: 600; }
.lead-table td { padding: 12px; border-bottom: 1px solid var(--border-color); color: var(--text-color); vertical-align: middle; }
.avatar { width: 40px; height: 40px; min-width: 40px; border-radius: 50%; background-color: #2E86C1; color: white !important; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; font-size: 16px; margin-right: 15px; }
.name-cell { display: flex; align-items: center; }
.status-badge { background-color: #27AE60; color: white !important; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; }
.view-btn { background-color: #2E86C1; color: white !important; padding: 8px 16px; text-decoration: none; border-radius: 4px; font-size: 14px; font-weight: 500; display: inline-block; text-align: center; }
.view-btn:hover { background-color: #1E3A5F; }
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
                html_content += '''<tr><td colspan="4" style="text-align: center; padding: 20px; color: gray;">No profiles match the selected filters.</td></tr>'''
            else:
                for _, row in filtered_df.iterrows():
                    name = str(row['Name']).replace("<", "&lt;").replace(">", "&gt;")
                    initials = "".join([n[0] for n in name.split() if n.isalnum()])[:2].upper() if name and name != "N/A" else "??"
                    title = str(row['Job Title']).replace("<", "&lt;").replace(">", "&gt;")
                    url = str(row['LinkedIn URL']).replace('"', '&quot;')
                    
                    html_content += f'''
<tr>
<td><div class="name-cell"><div class="avatar">{initials}</div><strong>{name}</strong></div></td>
<td>{title}</td>
<td><span class="status-badge">● Active</span></td>
<td><a href="{url}" target="_blank" class="view-btn">View Profile</a></td>
</tr>'''
                    
            html_content += "</tbody>\n</table>"
            st.markdown(html_content, unsafe_allow_html=True)

    # --- Feature 4: Automated Job Posting ---
    with tab4:
        st.header("Automated Job Posting")
        st.markdown("Generate a professional job posting dynamically tailored to your region.")
        
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g. Senior Data Scientist")
            company_name = st.text_input("Company Name", placeholder=cd["companies"])
            location = st.text_input("Location", value=f"{selected_city}, {selected_country}")
        with col2:
            req_skills = st.text_input("Required Skills", placeholder="e.g. Python, SQL, Machine Learning")
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
                
                # --- Post This Job Section ---
                st.markdown("---")
                st.markdown("### 📢 Post This Job")
                
                if selected_country == "Pakistan":
                    platforms = [
                        ("LinkedIn", "https://www.linkedin.com/job-posting/"),
                        ("Rozee.pk", "https://employer.rozee.pk/"),
                        ("Mustakbil.com", "https://www.mustakbil.com/employers"),
                        ("Indeed Pakistan", "https://pk.indeed.com/hire")
                    ]
                elif selected_country in ["USA", "UK", "Canada", "Australia"]:
                    platforms = [
                        ("LinkedIn", "https://www.linkedin.com/job-posting/"),
                        ("Indeed", "https://www.indeed.com/hire"),
                        ("Glassdoor", "https://www.glassdoor.com/post-job"),
                        ("Monster", "https://hiring.monster.com/")
                    ]
                elif selected_country in ["UAE", "Saudi Arabia"]:
                    platforms = [
                        ("LinkedIn", "https://www.linkedin.com/job-posting/"),
                        ("Bayt.com", "https://www.bayt.com/en/employer/"),
                        ("Indeed Gulf", "https://ae.indeed.com/hire"),
                        ("Naukrigulf", "https://www.naukrigulf.com/employer/")
                    ]
                else:
                    platforms = [
                        ("LinkedIn", "https://www.linkedin.com/job-posting/"),
                        ("Indeed", "https://www.indeed.com/hire")
                    ]
                
                cols = st.columns(len(platforms))
                for idx, (plat_name, plat_url) in enumerate(platforms):
                    with cols[idx]:
                        # Using anchor tag styled as a button
                        st.markdown(f'<a href="{plat_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; background-color:#2E86C1; color:white; border:none; padding:10px; border-radius:5px; cursor:pointer;">{plat_name}</button></a>', unsafe_allow_html=True)

            else:
                st.warning("Please enter at least the Job Title and Company Name.")

if __name__ == "__main__":
    main()
