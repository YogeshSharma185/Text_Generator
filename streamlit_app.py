import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ---------------- MODELS ---------------- #

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class TransformerTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=2, max_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.transformer(x)
        return self.fc(x[:, -1])


# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_models():
    lstm_ckpt = torch.load("lstm_model.pth", map_location="cpu")
    transformer_ckpt = torch.load("transformer_model.pth", map_location="cpu")

    lstm_model = LSTMTextGenerator(len(lstm_ckpt["word2idx"]))
    transformer_model = TransformerTextGenerator(len(transformer_ckpt["word2idx"]))

    lstm_model.load_state_dict(lstm_ckpt["model_state"])
    transformer_model.load_state_dict(transformer_ckpt["model_state"])

    lstm_model.eval()
    transformer_model.eval()

    return (
        lstm_model,
        transformer_model,
        lstm_ckpt["word2idx"],
        lstm_ckpt["idx2word"]
    )


# ---------------- TEXT GENERATION ---------------- #

def generate_text(model, prompt, word2idx, idx2word, length, temperature, seq_len=20):
    words = prompt.lower().split()
    current = [word2idx.get(w, 0) for w in words][-seq_len:]

    for _ in range(length):
        x = torch.tensor([current])
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs /= probs.sum()

        idx = np.random.choice(len(probs), p=probs)
        words.append(idx2word.get(idx, ""))
        current.append(idx)
        current = current[-seq_len:]

    return " ".join(words)


# ---------------- UI ---------------- #

st.set_page_config(layout="wide")

st.title("🧠 AI Text Generator – LSTM & Transformer")
st.write("Generate text using LSTM, Transformer, or compare both side-by-side.")

prompt = st.text_input("Enter prompt", "the night was")
length = st.slider("Number of words to generate", 10, 100, 40)
temperature = st.slider("Creativity (temperature)", 0.5, 1.2, 0.8)

# with st.expander("ℹ️ What is Temperature?"):
#     st.write("""
# - **Low (0.5)** → safer, repetitive text  
# - **Medium (0.7–0.9)** → balanced creativity  
# - **High (1.0+)** → highly creative, less grammatical  
# """)
with st.expander("🔍 What is Temperature in Text Generation?"):
    st.markdown("""
    **Temperature controls randomness in text generation.**

    - 🔹 **Low Temperature (0.2 – 0.5)**  
      - Very confident & repetitive text  
      - Predicts most probable next word  
      - Less creativity

    - 🔹 **Medium Temperature (0.6 – 0.8)**  
      - Balanced output  
      - Grammatically correct + creative  
      - **Recommended**

    - 🔹 **High Temperature (0.9 – 1.2)**  
      - More randomness  
      - Creative but may lose meaning  
      - Can produce unexpected words

    **In this project:**  
    Temperature is applied using **softmax scaling** before sampling the next word.
    """)

lstm_model, transformer_model, word2idx, idx2word = load_models()

# ---------------- BUTTONS ---------------- #

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    lstm_btn = st.button("🔵 Generate with LSTM")

with col_btn2:
    transformer_btn = st.button("🟢 Generate with Transformer")

with col_btn3:
    compare_btn = st.button("⚖️ Generate & Compare")

# ---------------- OUTPUT ---------------- #

if lstm_btn:
    st.subheader("🔵 LSTM Output")
    text = generate_text(
        lstm_model, prompt, word2idx, idx2word, length, temperature
    )
    st.write(text)

elif transformer_btn:
    st.subheader("🟢 Transformer Output")
    text = generate_text(
        transformer_model, prompt, word2idx, idx2word, length, temperature
    )
    st.write(text)

elif compare_btn:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 LSTM Output")
        lstm_text = generate_text(
            lstm_model, prompt, word2idx, idx2word, length, temperature
        )
        st.write(lstm_text)

    with col2:
        st.subheader("🟢 Transformer Output")
        transformer_text = generate_text(
            transformer_model, prompt, word2idx, idx2word, length, temperature
        )
        st.write(transformer_text)
