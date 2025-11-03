import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="News Bias Detector",
    page_icon="📰",
    layout="wide"
)

# ==================== LOAD MODELS & ARTIFACTS ====================

@st.cache_resource
def load_artifacts():
    """Load all saved models and artifacts"""
    try:
        # Load CNN model
        model = load_model('bias_detector_model.h5')
        
        # Load vocabulary
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load config
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        return model, vocab, label_encoder, config
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run train_model.py first to train and save the model.")
        st.stop()

# ==================== TEXT PREPROCESSING ====================

def clean_text(text):
    """Clean and normalize text (same as training)"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Simple tokenization"""
    return text.split()

def text_to_sequence(text, vocab):
    """Convert text to sequence of integers"""
    tokens = tokenize_text(clean_text(text))
    sequence = [vocab.get(token, 0) for token in tokens]
    return sequence

# ==================== BIAS LABEL MAPPING ====================

def get_bias_info(label_value):
    """Get bias category information based on 0-4 scale"""
    bias_mapping = {
        0: {
            'name': 'Strongly Left',
            'color': '#1E40AF',
            'description': 'Strongly left-leaning perspective'
        },
        1: {
            'name': 'Left',
            'color': '#3B82F6',
            'description': 'Left-leaning perspective'
        },
        2: {
            'name': 'Center',
            'color': '#6B7280',
            'description': 'Neutral/centrist perspective'
        },
        3: {
            'name': 'Right',
            'color': '#EF4444',
            'description': 'Right-leaning perspective'
        },
        4: {
            'name': 'Strongly Right',
            'color': '#991B1B',
            'description': 'Strongly right-leaning perspective'
        }
    }
    
    return bias_mapping.get(label_value, bias_mapping[2])

# ==================== PREDICTION ====================

def predict_bias(text, model, vocab, label_encoder, max_length):
    """Predict bias for input text"""
    # Preprocess text
    sequence = text_to_sequence(text, vocab)
    
    # Pad sequence
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    
    # Predict
    predictions = model.predict(padded_sequence, verbose=0)
    
    # Get predicted class and probabilities
    predicted_idx = np.argmax(predictions[0])
    predicted_label_encoded = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = predictions[0][predicted_idx] * 100
    
    # Get all probabilities
    probabilities = {}
    for i in range(len(predictions[0])):
        label_encoded = label_encoder.inverse_transform([i])[0]
        bias_info = get_bias_info(label_encoded)
        probabilities[bias_info['name']] = predictions[0][i] * 100
    
    return predicted_label_encoded, confidence, probabilities

# ==================== VISUALIZATION ====================

def create_bias_gauge(confidence, bias_level):
    """Create a gauge chart for bias prediction"""
    bias_info = get_bias_info(bias_level)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': f"Confidence: {bias_info['name']}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bias_info['color']},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_probability_bars(probabilities):
    """Create horizontal bar chart for all probabilities"""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Color mapping based on bias scale
    color_map = {
        'Strongly Left': '#1E40AF',
        'Left': '#3B82F6',
        'Center': '#6B7280',
        'Right': '#EF4444',
        'Strongly Right': '#991B1B'
    }
    
    colors = [color_map.get(label, '#6B7280') for label in labels]
    
    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Bias Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="Bias Category (0-4 Scale)",
        height=400,
        showlegend=False
    )
    
    return fig

# ==================== MAIN APP ====================

def main():
    # Load artifacts
    model, vocab, label_encoder, config = load_artifacts()
    
    # Header
    st.title("📰 News Bias Detector")
    st.markdown("### Deep Learning-based Political Bias Classification")
    st.markdown("*Using CNN + Word2Vec embeddings trained on political bias dataset*")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses deep learning to detect political bias in text on a scale from 0-4.
        
        **Model Architecture:**
        - Word2Vec embeddings (100D)
        - CNN with max pooling
        - Dense classification layers
        
        **Bias Scale (0-4):**
        """)
        
        # Display bias scale
        for i in range(5):
            bias_info = get_bias_info(i)
            st.markdown(f"- **{i}**: {bias_info['name']} - {bias_info['description']}")
        
        st.markdown("---")
        st.markdown(f"**Vocabulary Size:** {config['vocab_size']:,}")
        st.markdown(f"**Max Sequence Length:** {config['max_length']}")
        st.markdown(f"**Embedding Dimensions:** {config['embedding_dim']}")
        st.markdown(f"**Number of Classes:** {config['num_classes']}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter News Text or Statement")
        
        # Text input
        user_input = st.text_area(
            "Paste a news article, statement, or any political text:",
            height=250,
            placeholder="Enter text to analyze for political bias..."
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Bias", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Model Status", "Ready", "✓")
        st.metric("Processing", "Real-time", "⚡")
        st.metric("Bias Scale", "0-4", "📊")
    
    # Prediction
    if analyze_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Predict
                predicted_label, confidence, probabilities = predict_bias(
                    user_input, model, vocab, label_encoder, config['max_length']
                )
                
                # Get bias info
                bias_info = get_bias_info(predicted_label)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main result
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### Bias Rating: **{predicted_label}** - {bias_info['name']}")
                    st.markdown(f"*{bias_info['description']}*")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    # Confidence indicator
                    if confidence > 75:
                        st.success("High")
                    elif confidence > 50:
                        st.info("Medium")
                    else:
                        st.warning("Low")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gauge chart
                    gauge_fig = create_bias_gauge(confidence, predicted_label)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    # Probability bars
                    bar_fig = create_probability_bars(probabilities)
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                # Detailed probabilities
                with st.expander("📈 View Detailed Probabilities"):
                    st.markdown("**Probability for each bias rating:**")
                    for i in range(5):
                        info = get_bias_info(i)
                        prob = probabilities.get(info['name'], 0)
                        st.progress(float(prob) / 100, text=f"{i} - {info['name']}: {prob:.2f}%")
                
                # Text analysis
                with st.expander("📝 Text Analysis"):
                    tokens = tokenize_text(clean_text(user_input))
                    st.markdown(f"**Word Count:** {len(user_input.split())}")
                    st.markdown(f"**Processed Tokens:** {len(tokens)}")
                    st.markdown(f"**Character Count:** {len(user_input)}")
                    
                    # Show some processed tokens
                    if len(tokens) > 0:
                        st.markdown(f"**First 10 Tokens:** {', '.join(tokens[:10])}")
                
                # Disclaimer
                st.info("""
                **Disclaimer:** This is an AI-based prediction model. Results should be interpreted 
                as indicators rather than definitive classifications. Political bias detection is 
                complex and context-dependent. Use results as one of many tools for media literacy.
                """)

if __name__ == "__main__":
    main()