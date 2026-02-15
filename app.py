import streamlit as st
import cv2
import numpy as np
import easyocr
import google.generativeai as genai
import pandas as pd
import plotly.express as px
from PIL import Image
import os
from dotenv import load_dotenv
import json

# 1. Configuration & Setup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="AI Receipt Analyzer", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. Functions for the Logical Pipeline
def preprocess_image(image):
    """Applies grayscale and thresholding to improve OCR accuracy."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def run_ocr(processed_img):
    """Extracts raw text using EasyOCR."""
    reader = easyocr.Reader(['en'])
    result = reader.readtext(processed_img, detail=0)
    return " ".join(result)

def analyze_with_llm(raw_text):
    """Uses Gemini to structure data and provide financial advice."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Analyze this raw OCR text from a receipt: '{raw_text}'
    
    1. Clean the items and prices into a structured list.
    2. Categorize each item (e.g., Food, Groceries, Transport, Misc).
    3. Calculate the total.
    4. Provide 2-3 sentences of personalized budgeting advice based on these items.

    Return ONLY a JSON object with this exact structure:
    {{
        "items": [
            {{"item": "name", "price": 0.00, "category": "category"}}
        ],
        "total": 0.00,
        "advice": "your advice here"
    }}
    """
    
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

# 3. Streamlit UI
st.title("🧾 AI Receipt Analyzer & Financial Advisor")
st.caption("“You don't have to see the whole staircase, just take the first step.”")

uploaded_file = st.file_uploader("Upload a receipt (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display Layout
    col1, col2 = st.columns([1, 1.5])
    
    original_image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("🖼️ Uploaded Receipt")
        st.image(original_image, use_container_width=True)

    with st.spinner("🧠 Processing Pipeline: Image -> OCR -> AI Analysis..."):
        # Step 1: Preprocess
        processed_img = preprocess_image(original_image)
        
        # Step 2: OCR
        raw_text = run_ocr(processed_img)
        
        # Step 3: LLM Analysis
        try:
            data = analyze_with_llm(raw_text)
            
            with col2:
                st.subheader("📊 Spending Insights")
                
                # Metrics
                st.metric("Total Spent", f"${data['total']:.2f}")
                
                # Data Table
                df = pd.DataFrame(data['items'])
                st.dataframe(df, use_container_width=True)
                
                # Chart
                fig = px.pie(df, values='price', names='category', title="Spending by Category", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Advice
                st.info(f"💡 **AI Financial Advice:** {data['advice']}")
                
        except Exception as e:
            st.error(f"Analysis failed. Ensure your API key is correct. Error: {e}")

else:
    st.info("👋 Welcome! Please upload a receipt image to begin the analysis.")