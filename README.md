# 🧠 AI Text Generator – LSTM & Transformer

An end-to-end **AI Text Generation system** built using **deep learning language models**.  
This project allows users to generate creative text using **LSTM** and **Transformer** models, compare their outputs side-by-side, and control creativity using **temperature sampling**.

---

## 🚀 Features

- 📓 **Model training in Jupyter Notebook**
- 🧠 Two language models:
  - LSTM-based Language Model
  - Transformer-based Language Model
- 📊 Model evaluation:
  - Training loss curve
  - Model perplexity
- 🎛 Creativity control using **Temperature**
- 🖥 Streamlit Web App with:
  - Generate using **LSTM**
  - Generate using **Transformer**
  - **Side-by-side comparison**
- ⏱ Training progress display:
  - Epoch number
  - Time taken
  - Estimated remaining time

---

## 📁 Project Structure

```text
AI-Text-Generator/
│
├── data/
│   ├── text.txt
│   
│
│── lstm_model.pth
│── transformer_model.pth
│
├── streamlit_app.py
├── Text_Generation.ipynb
├── requirements.txt
└── README.md
```


---

## 📚 Dataset Used

**Sherlock Holmes stories (public domain)**

### Why Sherlock Holmes?
✔ Rich English vocabulary  
✔ Long coherent paragraphs  
✔ Narrative structure (ideal for language modeling)  
✔ Public-domain (safe to use)

### Limitations
- Model tends to generate **story-like / novel-style text**
- Not ideal for:
  - Question answering
  - Factual responses
  - Chat-style conversations

👉 You can train on **any text dataset** (news, blogs, conversations) to change behavior.

---

## 🧪 Model Evaluation

### 🔹 Training Loss
- Loss **decreases with epochs**, indicating proper learning
- Visualized using a loss curve

### 🔹 Perplexity
Example:
Model Perplexity: 1.37


**What does this mean?**
- Perplexity measures how well the model predicts the next word
- Lower = better
- 1.37 means:
  > The model is highly confident and has learned strong word patterns

---

## 🎛 Temperature (Creativity Control)

Temperature controls **randomness during word sampling**.

| Temperature | Behavior |
|------------|---------|
| 0.5 | Safe, repetitive, predictable |
| 0.8 | Balanced creativity |
| 1.0+ | Highly creative but noisy |

Used formula:
log(probabilities) / temperature


Higher temperature → more randomness  
Lower temperature → safer output

---

## 🖥 Streamlit App UI

### Available Buttons

1️⃣ **Generate using LSTM**  
2️⃣ **Generate using Transformer**  
3️⃣ **Side-by-Side Comparison**

Users can:
- Enter a custom prompt
- Choose number of words
- Adjust creativity
- Compare outputs instantly

---

## 🔄 LSTM vs Transformer Comparison

| Feature | LSTM | Transformer |
|------|------|------------|
| Sequential memory | ✅ | ❌ |
| Long-range context | ❌ | ✅ |
| Training speed | Faster | Slower |
| Output quality | Good | Better |
| Parallelism | ❌ | ✅ |

---

## ⏱ Training Progress Display

During training, the notebook shows:
- Current epoch
- Total epochs
- Time taken per epoch
- Estimated remaining time (ETA)

This makes training transparent and trackable.

---

## 🛠 Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook
- Streamlit

---

## ▶ How to Run

### 1️⃣ Install Dependencies
pip install -r requirements.txt


### 2️⃣ Train Models
Open and run:
- `Text_Generation.ipynb`

### 3️⃣ Launch Web App
streamlit run streamlit_app.py


---

## 🧑‍💻 Author

**Yogesh Sharma**  
AI / Machine Learning Engineer  
Built as a hands-on deep learning project to understand **sequence modeling, evaluation, and deployment**

---

