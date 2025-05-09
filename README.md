# KrushiMitra: Farmer Subsidy Assistant

<p align="center">
  <img src="https://via.placeholder.com/200x200.png?text=KrushiMitra" alt="KrushiMitra Logo" width="200"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-orange" alt="Version 1.0.0"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"/>
  <img src="https://img.shields.io/badge/platform-Web-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28-FF4B4B" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Gemini Flash 2.0-API-fbbc04" alt="Gemini API"/>
</p>

> Empowering Indian farmers with simplified access to government subsidies through intelligent conversation, organized documentation, and comparative analysis.

## 🌾 About The Project

**KrushiMitra** is a digital assistant that helps farmers navigate the complex landscape of agricultural subsidies. By providing personalized guidance, document checklists, and scheme comparisons in an accessible format, KrushiMitra aims to increase subsidy utilization among India's farming communities.

### 🎯 The Problem We Solve

- **Information Gap**: Many farmers lack awareness about available subsidies
- **Complex Documentation**: Unclear requirements lead to application rejections
- **Decision Difficulties**: Choosing between multiple schemes is challenging
- **Access Barriers**: Rural farmers face difficulties visiting government offices repeatedly

## ✨ Key Features

### 1️⃣ Intelligent Subsidy Chatbot
Conversational AI that answers farmers' questions about subsidies in simple language, available in multiple Indian languages.

### 2️⃣ Document Checklist Generator
Creates personalized lists of required documents based on eligible schemes, prioritized by importance and difficulty.

### 3️⃣ Scheme Comparison Tool
Side-by-side comparison of benefits, eligibility, and application complexity for multiple subsidy schemes.

## 🖥️ Screenshots

<p align="center">
  <img src="https://via.placeholder.com/250x500.png?text=Chatbot" alt="Chatbot Screenshot" width="250"/>
  <img src="https://via.placeholder.com/250x500.png?text=Document+Checklist" alt="Document Checklist Screenshot" width="250"/>
  <img src="https://via.placeholder.com/250x500.png?text=Scheme+Comparison" alt="Scheme Comparison Screenshot" width="250"/>
</p>

## 🔧 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Google API key for Gemini Flash 2.0

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/yourusername/krushimitra.git
   ```

2. Install Python dependencies
   ```sh
   cd krushimitra
   pip install -r requirements.txt
   ```

3. Set up environment variables
   ```sh
   # Create .env file with your API keys
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   echo "QDRANT_URL=your_qdrant_url_here" >> .env
   ```

4. Run the Streamlit app
   ```sh
   streamlit run app.py
   ```

## 📱 Deployment

### Streamlit Cloud
1. Fork this repository to your GitHub account
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository
4. Add the required secrets in the Streamlit Cloud dashboard

### Docker Deployment
```sh
docker build -t krushimitra .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key krushimitra
```

## 🗺️ Roadmap

- [x] Initial release with core features
- [ ] Integration with direct application submission
- [ ] Document digitization assistance
- [ ] Subsidy tracking notifications
- [ ] Community support forums
- [ ] AI-powered yield prediction based on scheme adoption

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
