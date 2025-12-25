# ProGAN Face Generator 🎭

<img width="1099" height="703" alt="Screenshot 2025-12-25 182923" src="https://github.com/user-attachments/assets/70f65f65-5b55-4a81-a585-35a09a854944" />

**An AI-powered high-resolution human face generator built with TensorFlow and ProGAN architecture.**

This project implements a **Progressive Growing GAN (ProGAN)** inspired architecture to generate realistic 128x128 human faces. Trained on the **CelebA dataset**, it utilizes advanced training techniques for stability and visual fidelity.

## 🚀 Key Features

* **Progressive Architecture:** Generates faces starting from low-res to high-res (128x128).
* **WGAN-GP Loss:** Uses Wasserstein Loss with Gradient Penalty for superior training stability (no more mode collapse!).
* **Perceptual Loss:** Integrated **VGG16 feature extraction** to ensure the generated faces look "natural" to human eyes.
* **Multi-GPU Support:** Fully optimized with `tf.distribute.MirroredStrategy` for dual-GPU training (specifically tuned for Kaggle/Colab T4 GPUs).
* **Mixed Precision:** Uses `mixed_float16` for 2x faster training and lower memory footprint.
* **PixelWise Normalization:** Prevents signal magnitude explosion during training.

## 🛠️ Tech Stack

* **Framework:** TensorFlow 2.16+ / Keras
* **Dataset:** CelebA (Faces)
* **Architecture:** ProGAN (Generator & Discriminator)
* **Deployment:** Streamlit
* **Hardware:** Dual NVIDIA T4 GPUs

## 📋 Project Workflow

1. **Data Pipeline:** High-speed globbing and multithreaded `tf.data` pipeline with prefetching.
2. **Custom Layers:** Implementation of `PixelNormalization` and `MiniBatchStd` for high-quality GAN training.
3. **Training:** * WGAN-GP for the adversarial loss.
* VGG16 based Perceptual loss for sharp details.
* Adam optimizer with custom learning rates ().


4. **Deployment:** Model weights exported via `pickle` and served through a Streamlit web interface.

## 💻 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/progan-face-generator.git
cd progan-face-generator

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the App

To see the generator in action, run:

```bash
streamlit run app.py

```

## 📊 Training Results

The model was trained for **50+ epochs** on 200k+ images.

* **Resolution:** 128x128 pixels.
* **Training Time:** Optimized using Dual T4 GPUs.
* **Checkpoints:** Periodic saving to prevent progress loss.

---

### Future Improvements

* [ ] Implement 256x256 and 512x512 scaling.
* [ ] Add Latent Space Interpolation (walking through faces).
* [ ] Integrate Style-based mapping (StyleGAN features).

**Developer:** [Your Name/Handle]

**App Link:** [Live Demo](https://progan-face-generator.streamlit.app/)
