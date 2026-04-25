import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, date, time, timedelta, timezone

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
    
    # --- Sidebar Global Settings ---
    st.sidebar.header("🌍 Worldwide Localization")
    
    # Step 1: Country Selection
    selected_country = st.sidebar.selectbox("Select Country", list(COUNTRY_DATA.keys()))
    
    # Step 2: City Selection based on Country
    cd = COUNTRY_DATA[selected_country]
    selected_city = st.sidebar.selectbox("Select Major City", cd["cities"])
    
    # --- Global Dashboard KPIs ---
    st.subheader("📊 Recruitment Overview")
    col1, col2, col3 = st.columns(3)
    
    total_scanned = 0
    top_score = "0%"
    avg_score = "0%"
    
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        df = st.session_state['results_df']
        total_scanned = len(df)
        top_score = f"{df.iloc[0]['Match Score (%)']}%"
        avg_score = f"{round(df['Match Score (%)'].mean(), 2)}%"
        
    col1.metric("Total Resumes Scanned", total_scanned)
    col2.metric("Top Match Score", top_score)
    col3.metric("Avg Match Score", avg_score)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📝 Job Posting Generator", "📊 Resume Screening", "📅 Interview Scheduling"])
    
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
            st.dataframe(
                df.style.background_gradient(cmap='Greens', subset=['Match Score (%)']),
                use_container_width=True
            )
            
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
                
                st.subheader("📅 Generated Interview Schedule")
                st.table(schedule_df)
                st.success("Interview schedule generated successfully!")
                
        else:
            st.info("Please scan and match resumes in the 'Resume Screening' tab first to generate a schedule.")

if __name__ == "__main__":
    main()
