import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Reshape, UpSampling2D, Conv2D, LeakyReLU
import numpy as np
from PIL import Image
import pickle
import os

# --- 1. Custom Layer (PixelNormalization) ---
class PixelNormalization(Layer):
    def __init__(self, epsilon=1e-7, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)

# --- 2. Build Architecture ---
def build_generator_final(noise_dim=128):
    model = tf.keras.Sequential([
        Input(shape=(noise_dim,)),
        Dense(4 * 4 * 512, use_bias=False),
        Reshape((4, 4, 512)),
        LeakyReLU(0.2),
        PixelNormalization(),
        UpSampling2D(), # 8x8
        Conv2D(512, (3, 3), padding="same", use_bias=False),
        LeakyReLU(0.2),
        PixelNormalization(),
        UpSampling2D(), # 16x16
        Conv2D(256, (3, 3), padding="same", use_bias=False),
        LeakyReLU(0.2),
        PixelNormalization(),
        UpSampling2D(), # 32x32
        Conv2D(128, (3, 3), padding="same", use_bias=False),
        LeakyReLU(0.2),
        PixelNormalization(),
        UpSampling2D(), # 64x64
        Conv2D(64, (3, 3), padding="same", use_bias=False),
        LeakyReLU(0.2),
        PixelNormalization(),
        UpSampling2D(), # 128x128
        Conv2D(32, (3, 3), padding="same", use_bias=False),
        LeakyReLU(0.2),
        PixelNormalization(),
        Conv2D(3, (1, 1), padding="same", activation="tanh")
    ])
    return model

# --- 3. UI ---
st.set_page_config(page_title="AI Face Gen 2025", page_icon="👤", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 15px; height: 3.5em;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        color: white; font-weight: bold; border: none;
    }
    .img-card { padding: 20px; background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. Loading via Pickle (Optimized for Cloud) ---
@st.cache_resource
def load_gan():
    pickle_path = 'generator_weights.pkl'
    
    if not os.path.exists(pickle_path):
        # Fallback for local testing if needed
        st.warning("⚠️ Weights file not found in root. Checking subdirectories...")
        return None
        
    try:
        model = build_generator_final()
        with open(pickle_path, 'rb') as f:
            weights = pickle.load(f)
        model.set_weights(weights)
        return model
    except Exception as e:
        st.error(f"❌ Load Error: {e}")
        return None

# --- 5. App UI ---
st.title("🎭 ProGAN Face Generator")
st.write("2025 AI Labs - Generate 128x128 faces of non-existent people.")

generator = load_gan()

if generator:
    st.divider()
    if st.button("✨ Generate AI Face"):
        with st.spinner("Wait, the AI is dreaming..."):
            # Noise generation
            noise = tf.random.normal((1, 128))
            
            # Prediction
            img_batch = generator(noise, training=False)
            
            # Post-Process (Index 0 for single image)
            img = img_batch[0].numpy()
            img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # PIL Image conversion
            final_img = Image.fromarray(img)
            
            # Display
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(final_img, width=400, caption="Generated AI Result")
            
            # Download Section
            temp_file = "face.png"
            final_img.save(temp_file)
            with open(temp_file, "rb") as f:
                st.download_button("📥 Download This Image", f, "ai_face.png", "image/png")
            st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Dec 2025 | Developed by Umar Farooq")
