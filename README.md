




Link to My model file click on the Below link to Download✅ model file 
1)Below link contain mental_health_model.pth🥶
https://drive.google.com/file/d/14UBuAH0vsNuKU_HLTTGg5RB1JnHzTfXS/view?usp=share_link
2)Link to other files:tokenizer/config.json/vocab.txt
https://drive.google.com/drive/folders/1m-k2J8hgm4MXdNcmulip60mu6rU6QTcw?usp=share_link

have a nice day hope you loved the work if you loved it then mention my username to others so others can also learn and create 
for furthur enquiry email at yashthakur1700@gmail.com


# Mental Health Detection System


AI-powered depression and stress detection using BERT + BiLSTM.

## Setup

1. **Download your trained model from Google Drive:**
   - `mental_health_model.pth`
   - `model_config.json`
   - `mental_health_tokenizer/` folder
   
   Place them in a folder called `models/`

2. **Install dependencies:**


http://localhost:8000

# Create .gitignore
touch .gitignore


# 🧠 Mental Health Detection System

AI-powered depression and stress detection using **BERT + BiLSTM** with a calming, animated frontend interface.

## 🎯 Project Overview

This application uses advanced Natural Language Processing (NLP) to detect signs of depression and stress from text input. It combines:
- **BERT** embeddings for contextual understanding
- **BiLSTM** for capturing sequential patterns
- **Attention mechanism** for focusing on important linguistic markers

## 🌟 Features

- ✅ Real-time mental health assessment from text
- ✅ Beautiful, calming UI with breathing animations
- ✅ Confidence scores and risk level classification
- ✅ Support resources for users in need
- ✅ FastAPI backend with RESTful endpoints
- ✅ SDG 3: Good Health and Well-Being

## 🏗️ Architecture



- **BERT**: Contextual word embeddings (768-dim)
- **BiLSTM**: Bidirectional LSTM (256 hidden units, 2 layers)
- **Attention**: Weighted feature extraction
- **Output**: Binary classification (Depression/Non-Depression)

## 📊 Model Performance

- **Accuracy**: [Your accuracy]%
- **F1 Score**: [Your F1 score]
- **ROC-AUC**: [Your ROC-AUC]
- **Dataset**: Reddit Mental Health Corpus

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
git clone https://github.com/YOUR_USERNAME/mental-health-detection.git
cd mental-health-detection

2. **Create virtual environment**

python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Download the trained model**

Due to file size limitations, model files are not included in the repository. 

**Option A**: Train your own model using the notebook provided
**Option B**: Download pre-trained models from [Google Drive link]

Place the following files in the `models/` directory:

models/
├── mental_health_model.pth
├── model_config.json
└── mental_health_tokenizer/


5. **Run the application**

uvicorn app:app --host 0.0.0.0 --port 8000

6. **Open your browser**
http://localhost:8000

## 📁 Project Structure


mental-health-detection/
├── app.py # FastAPI backend
├── index.html # Frontend with animations
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
├── README.md # Documentation
└── models/ # Model files (not in repo)
├── mental_health_model.pth
├── model_config.json
└── mental_health_tokenizer/


## 🔌 API Endpoints

### Health Check

GET /health

### Prediction

POST /predict
Content-Type: application/json

{
"text": "I feel so hopeless and empty..."
}


**Response:**



{
"prediction": "Depression/Stress Detected",
"confidence": 0.87,
"risk_level": "High",
"probabilities": {
"non_depression": 0.13,
"depression": 0.87
}
}


## 🧪 Testing

Test with sample statements:


Test depression detection
curl -X POST http://localhost:8000/predict
-H "Content-Type: application/json"
-d '{"text": "I feel hopeless and nothing makes me happy anymore"}'

Test positive statement
curl -X POST http://localhost:8000/predict
-H "Content-Type: application/json"
-d '{"text": "Had a wonderful day! Feeling blessed and grateful"}'


## 🎨 Frontend Features

- **Breathing Animation**: Calming circular animation to reduce anxiety
- **Smooth Transitions**: Gentle animations throughout the interface
- **Color Psychology**: Soothing blues and purples for mental wellness
- **Resource Links**: Direct access to mental health support services

## 🆘 Support Resources

- **KIRAN Mental Health**: 1800-599-0019 (24/7)
- **Vandrevala Foundation**: 1860-2662-345
- **iCall**: 9152987821 (Mon-Sat, 8 AM - 10 PM)
- **NIMHANS**: +91 80 4611 0007

## 🔬 Technology Stack

- **Backend**: FastAPI, Python 3.10
- **ML Framework**: PyTorch 2.9.0
- **NLP**: Transformers (Hugging Face)
- **Model**: BERT + BiLSTM + Attention
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Uvicorn ASGI Server

## 📈 Future Improvements

- [ ] Multi-language support
- [ ] Real-time chat analysis
- [ ] Mobile app version
- [ ] Integration with therapy platforms
- [ ] Continuous model retraining
- [ ] User feedback loop

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This tool is for informational purposes only and should not replace professional mental health diagnosis or treatment. If you're experiencing mental health concerns, please consult a qualified healthcare professional.


## 👨‍💻 Author

**Yash Pratap Singh**
- GitHub: [@YASH7110](https://github.com/YASH7110)
- Email: yashthakur1700@gmail.com

## 🙏 Acknowledgments

- Reddit Mental Health Corpus dataset
- Hugging Face Transformers library
- SDG 3: Good Health and Well-Being initiative

## 📚 References

1. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers
2. Reddit Mental Health Dataset (Kaggle)
3. CLPsych Workshop on Computational Linguistics and Clinical Psychology

---

Made with ❤️ for mental health awareness
