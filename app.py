import streamlit as st
import pickle
import re
from PyPDF2 import PdfReader

# Load models
rf_classifier_categorization = pickle.load(open('_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Clean resume function
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to predict category
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Function to predict job recommendation
def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

# Function to extract text from PDF
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Functions to extract information from resume
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_skills_from_resume(text):
    skills_list = [ 'Python', 'Data Analysis', 'Machine Learning', 'SQL', 'Tableau', 'Java', 'C++', 'JavaScript',
                    'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'Git', 'Matplotlib', 'Seaborn', 'Numpy', 'Pandas']
    skills = [skill for skill in skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    return skills

def extract_education_from_resume(text):
    education_keywords = ['Computer Science', 'Information Technology', 'Software Engineering', 'Business Administration', 'Marketing']
    education = [edu for edu in education_keywords if re.search(rf"\b{re.escape(edu)}\b", text, re.IGNORECASE)]
    return education

def extract_name_from_resume(text):
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    return match.group() if match else None

# Streamlit App UI
st.title("Resume Analyzer and Job Recommender")

uploaded_file = st.file_uploader("Upload a PDF or TXT resume", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        resume_text = pdf_to_text(uploaded_file)
    elif uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode('utf-8')

    # # Analyze resume
    predicted_category = predict_category(resume_text)
    recommended_job = job_recommendation(resume_text)
    phone = extract_contact_number_from_resume(resume_text)
    email = extract_email_from_resume(resume_text)
    extracted_skills = extract_skills_from_resume(resume_text)
    extracted_education = extract_education_from_resume(resume_text)
    name = extract_name_from_resume(resume_text)

    # Display results
    st.subheader("Predicted Category")
    st.write(predicted_category)

    st.subheader("Recommended Job")
    st.write(recommended_job)

    st.subheader("Extracted Information")
    st.write(f"Name: {name}")
    st.write(f"Phone: {phone}")
    st.write(f"Email: {email}")

    st.subheader("Extracted Skills")
    st.write(", ".join(extracted_skills) if extracted_skills else "No skills extracted")

    st.subheader("Extracted Education")
    st.write(", ".join(extracted_education) if extracted_education else "No education details extracted")

else:
    st.write("Please upload a resume to analyze.")
