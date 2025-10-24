from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="Mental Health Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Model Definition
# ===========================

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=256, 
                 lstm_layers=2, num_classes=2, dropout=0.3):
        super(BERTBiLSTMClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def attention_net(self, lstm_output):
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = lstm_output * attention_weights
        return torch.sum(weighted_output, dim=1)
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        attended_output = self.attention_net(lstm_output)
        x = self.dropout(attended_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# ===========================
# Load Model on Startup with Detailed Progress
# ===========================

@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    
    print("\n" + "="*60)
    print("🔄 LOADING MENTAL HEALTH DETECTION MODEL")
    print("="*60)
    print("⏳ This may take 2-5 minutes on first run...")
    print("⏳ BERT model will be downloaded if not cached...")
    print()
    
    # Enhanced device detection for Mac
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
    
    print()
    
    # Load config
    config_path = 'models/model_config.json'
    print(f"📂 Step 1/4: Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"   ✅ Config loaded successfully")
    print(f"   Model: {config['bert_model_name']}")
    print(f"   Hidden dim: {config['hidden_dim']}")
    print(f"   LSTM layers: {config['lstm_layers']}")
    print()
    
    # Initialize model architecture
    print("🏗️  Step 2/4: Building model architecture...")
    print("   ⏳ Downloading BERT from HuggingFace (if needed)...")
    print("   This is a one-time download (~500MB)...")
    model = BERTBiLSTMClassifier(
        bert_model_name=config['bert_model_name'],
        hidden_dim=config['hidden_dim'],
        lstm_layers=config['lstm_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    print("   ✅ Model architecture built successfully!")
    print()
    
    # Load trained weights
    model_path = 'models/mental_health_model.pth'
    print(f"⚖️  Step 3/4: Loading trained weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("   ✅ Weights loaded successfully!")
    print()
    
    # Load tokenizer
    tokenizer_path = 'models/mental_health_tokenizer'
    print(f"📝 Step 4/4: Loading tokenizer from: {tokenizer_path}")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print("   ✅ Tokenizer loaded successfully!")
    print()
    
    print("="*60)
    print(f"🎉 MODEL LOADED SUCCESSFULLY ON {device.type.upper()}!")
    print("🚀 Ready to make predictions!")
    print("="*60)
    print()

# ===========================
# Request/Response Models
# ===========================

class TextInput(BaseModel):
    text: str

# ===========================
# API Routes
# ===========================

@app.get("/")
async def home():
    """Serve the frontend HTML"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "model": "loaded",
        "device": str(device)
    }

@app.post("/predict")
async def predict(input_data: TextInput):
    """Predict mental health status from text"""
    try:
        text = input_data.text
        
        if not text or len(text) < 10:
            raise HTTPException(
                status_code=400,
                detail="Text must be at least 10 characters long"
            )
        
        # Tokenize the input text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction using YOUR trained model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        pred_class = int(prediction.item())
        pred_label = "Depression/Stress Detected" if pred_class == 1 else "No Signs Detected"
        
        # Determine risk level
        if pred_class == 1:
            risk = "High" if confidence.item() > 0.8 else "Moderate"
        else:
            risk = "Low"
        
        return {
            "prediction": pred_label,
            "confidence": float(confidence.item()),
            "risk_level": risk,
            "probabilities": {
                "non_depression": float(probs[0][0]),
                "depression": float(probs[0][1])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# Run Server
# ===========================
# ===========================
# Run Server
# ===========================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Mental Health Detection API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("⏳ Please wait for model to load...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)  # Changed to False
