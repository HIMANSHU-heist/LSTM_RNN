# 🔮 Next Word Predictor using LSTM Neural Network

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://lstmrnn-predictbyhima.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **What comes next?** — An LSTM-powered next word prediction system. Type any sequence of words and watch the model predict what follows — just like your phone keyboard, but built from scratch.

---

## 🚀 Live Demo

🔗 **[Try it Live → lstmrnn-predictbyhima.streamlit.app](https://lstmrnn-predictbyhima.streamlit.app/)**

---

## 📌 Problem Statement

Simple RNNs suffer from the **Vanishing Gradient Problem** — gradients shrink to near-zero during backpropagation, making it impossible to learn long-range dependencies in text.

**LSTM (Long Short-Term Memory)** solves this with a gating mechanism:

| Gate | Function |
|---|---|
| 🚪 Forget Gate | "What old information should I erase?" |
| ✍️ Input Gate | "What new information should I store?" |
| 📤 Output Gate | "What should I output right now?" |

These gates give LSTM **selective long-term memory** — it remembers what matters and forgets what doesn't.

---

## 🏗️ Project Architecture

```
Raw Text Corpus
      ↓
Tokenization + N-gram Sequence Generation
      ↓
Padding (pre-padding)
      ↓
Embedding Layer
      ↓
LSTM(150) ← Long-term memory layer 1
      ↓
LSTM(100) ← Long-term memory layer 2
      ↓
Dense(vocab_size, Softmax)
      ↓
Predicted Next Word
```

---

## 📊 Dataset

- **Source:** Custom text corpus
- **Task:** Language Modeling — Predict next word
- **Approach:** N-gram sequence generation from raw text
- **Tokenizer:** Keras Tokenizer

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Preprocessing | Keras Tokenizer, Pad Sequences |
| Architecture | Stacked LSTM |
| Deployment | Streamlit |

---

## 🔬 What Happens Under the Hood

### 1. Sequence Generation
```python
# Convert text to n-gram sequences
for sentence in corpus:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# Pad sequences
sequences = pad_sequences(input_sequences, 
                          maxlen=max_len, 
                          padding='pre')
```

### 2. LSTM Architecture
```python
model = Sequential([
    Embedding(total_words, 100, input_length=max_len-1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

### 3. Prediction
```python
def predict_next_word(seed_text, model, tokenizer, max_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Architecture | Stacked LSTM (150 + 100 units) |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Output | Softmax over full vocabulary |

---

## 🛠️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/lstm-next-word-predictor
cd lstm-next-word-predictor

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
lstm-next-word-predictor/
│
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── model.h5            # Saved LSTM model
├── tokenizer.pkl       # Saved Keras tokenizer
└── requirements.txt    # Dependencies
```

---

## 🧠 Key Learnings

- Why simple RNNs fail — Vanishing Gradient Problem
- LSTM gating mechanism — Forget, Input, Output gates
- N-gram sequence generation for language modeling
- Stacked LSTM — deeper memory representation
- Softmax for multi-class word prediction
- The foundation of GPT and modern LLMs

---

## 🔗 Connection to Modern LLMs

```
LSTM (this project)
      ↓
Attention Mechanism
      ↓
Transformer Architecture
      ↓
GPT / LLaMA / Gemini

Everything started here. 🔥
```

---

## 👨‍💻 Author

**Himanshu Bendale**
- 🎓 B.E. AI & DS — Mumbai University
- 🎓 B.S. Electronic Systems — IIT Madras
- 🔗 [GitHub](https://github.com/yourusername)
- 💼 [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🗺️ Roadmap

```
✅ ANN  — Churn Prediction (deployed)
✅ RNN  — Sentiment Analysis (deployed)
✅ LSTM — Next Word Predictor (deployed)
🔜 Transformers — Attention Mechanism
🔜 LLM Fine-Tuning — LoRA/QLoRA on Llama 3
```

> *"Everything started here. The grind is on."* 💪
