# 🚀 AI-Driven Hyper-Personalization & Recommendation

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
In today's market, customers are no longer satisfied with one-size-fits-all products or generic services. With the rise of advanced technologies like generative AI, consumers have become accustomed to a level of personalization that tailors products, services, and experiences to their very specific preferences. This shift is transforming how businesses must engage with their customers.

## 🎥 Demo
🔗 [Live Demo](https://huggingface.co/spaces/Vaibhav84/RecommendationApp) (Live Demo Link Added)

📹 [Video Demo](https://youtu.be/bV7JA8PVZhk) (Video link added from youtube)  
🖼️ Screenshots: 1
[![Video Link](https://img.youtube.com/vi/bV7JA8PVZhk/hqdefault.jpg)](https://youtu.be/bV7JA8PVZhk)

![Screenshot 1](link-to-image)

## 💡 Inspiration
Unlock the future of shopping with AI that understands customer—not just customer's preferences, but their passions, their desires, and their dreams. Empowering customers with recommendations that feel as unique as they are—because personalization isn’t just a feature, it’s the foundation of meaningful connection.

## ⚙️ What It Does
Built an application with emerging AI capabilities which porvides a unique user experience that include different dashboards. These dashboards are of a wide range depending on customer's recent purchases, their shift from one product segment to another, based on multi-modal personalisation, sentiments, financial needs etc., Based on these factors, application would fetch the right suggestions to cater the customer needs.

## 🛠️ How We Built It
Application is build by utilizing the below techonologies:
1. Streamlit: For creating interactive web applications and the user to interact with Streamlit web appication
2. FastAPI: For building APIs to handle requests, responses and Streamlit sends requests to the FastAPI backend. The processed data and visualizations are sent back to the Streamlit web application for display.
3. Hugging Face: For NLP and other machine learning models and FastAPI may use Hugging Face models for NLP tasks
4. Python: The main programming language used for scripting and data processing. FastAPI processes the data using Python scripts
5. Amazon Q: For faster code development 
6. Excel As (Data Set): Excel files used as data sources. FastAPI reads data from Excel files using pandas
7. Sklearn: For machine learning algorithms. FastAPI uses Sklearn for machine learning tasks
8. Scipy: For scientific computing and sparse matrices. FastAPI uses Scipy for scientific computing
9. Dall-E: For image generation. FastAPI generates images using Dall-E
10. Gemini LLModel: For adaptive content recommendation. FastAPI uses Gemini LLModel for content recommendation
11. Plotly: For creating graphs and visualizations. FastAPI generates graphs and visualizations using Plotly
12. LLm Model - Phi : For Content generatio

## 🚧 Challenges We Faced
1. Selection of right LLM model in building the application is a major challenge
2. Fetching data dynamically based on customer transaction
3. Preparing and training sample data that would allow the application to run smoothly with required amount of data

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/aidhp-mistral.git
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project  
   ```sh
   for api
   uvicorn app:app --reload  # or for api
   for app
   streamlit run app.py # for app to run
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: Streamlit / Hugging Face
- 🔹 Backend: FastAPI / Python 
- 🔹 Database: Excel As / Json Files
- 🔹 Other: Amazon Q / Sklearn / Scipy / Dall-E / Gemini LLModel / Plotly / Nvidia / Microsoft / Phi-4-mini-imstruct

## 👥 Team - Mistral
Vaibhav Gupta 
Anil Kudala 
Santosh Parida 
Ashish Kumar 
Venkata Siva Sairam Puranapanda 

