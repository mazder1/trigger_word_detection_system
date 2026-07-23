<div align="center">

# 🎙️ Trigger Word Detection

**A deep learning wake-word detector — it listens to 10 seconds of audio and tells you *exactly when* someone said “activate”.**

The same idea behind *“Alexa”*, *“Hey Siri”* and *“OK Google”*, built from scratch with a 1-D convolution and two GRUs.

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Params](https://img.shields.io/badge/parameters-523%2C329-blue?style=flat-square)
![Dev accuracy](https://img.shields.io/badge/dev%20accuracy-83.0%25-brightgreen?style=flat-square)
![Epochs](https://img.shields.io/badge/epochs-500-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/status-reference%20implementation-orange?style=flat-square)

</div>

---

## 📌 About

Speech-activated devices need to answer one question continuously: *did the user just say the magic word?*

This project answers it end to end. It **synthesises its own training set** from raw audio fragments, converts each clip to a spectrogram, and trains a recurrent network to emit a probability **at every one of 1375 timesteps** — so the output isn’t just “yes, the word occurred”, it’s “the word ended *right here*”.

Built while working through **Andrew Ng’s Deep Learning Specialization**, *Course 5 — Sequence Models*, Week 3. The architecture and data pipeline follow the course’s trigger-word assignment; the implementation in this notebook is my own.

> [!IMPORTANT]
> **This repository contains code only — it will not run as-is.**
> The audio corpus (`raw_data/`), the pre-built dev set (`XY_dev/`) and the `td_utils.py` helper module belong to **DeepLearning.AI** and are not mine to redistribute, so they are deliberately not committed here. See [Reproducing this](#-reproducing-this) for how to supply them yourself. The notebook is published with **all outputs and figures intact**, so you can read the full training run without executing a single cell.

---

## 🧠 How it works

### 1 · Synthesising the dataset

Hand-labelling wake words is slow and expensive. Instead, every training example is **manufactured** by overlaying short word clips onto background noise — which means the labels come out perfectly aligned for free.

Each 10-second example is assembled from three ingredients:

| Ingredient | Length | Count per example | Role |
|:--|:--|:--|:--|
| 🔊 **Background noise** | 10 s | 1 | The canvas — attenuated by 20 dB |
| ✅ **“activate”** | ~1 s | 0 – 4 | The positive trigger |
| ❌ **Other words** | ~1 s | 0 – 2 | Negatives, to punish false alarms |

Clips are dropped at random offsets, with a collision check that retries up to 5 times so two words never overlap. The finished track is normalised to −20 dBFS and rendered to a spectrogram.

### 2 · Labelling the output

The label is a **time series, not a single number**. The moment an “activate” clip finishes, the next **50 timesteps** are marked `1`:

```text
  audio   ──────[ activate ]────────[ other word ]──────────[ activate ]──────
                              │                                        │
  labels  0000000000000000000011111111111100000000000000000000000000000111111
                              └── 50 steps ──┘
```

Why a *window* of ones rather than a single spike? A lone `1` in 1375 zeros is a needle in a haystack — the network would learn to predict all-zeros and still score 99.9% accuracy. Widening the target makes the positive class learnable.

### 3 · The network

```mermaid
flowchart LR
    A["🎧 Raw audio<br/>10 seconds"] --> B["📊 Spectrogram<br/>5511 × 101"]
    B --> C["Conv1D<br/>196 filters · k=15 · s=4"]
    C --> D["BatchNorm<br/>+ ReLU + Dropout"]
    D --> E["GRU 128<br/>return_sequences"]
    E --> F["GRU 128<br/>return_sequences"]
    F --> G["TimeDistributed<br/>Dense 1 · sigmoid"]
    G --> H["📈 1375 probabilities<br/>one per timestep"]

    style A fill:#1f6feb,stroke:#0d419d,color:#fff
    style B fill:#6e40c9,stroke:#4c2889,color:#fff
    style C fill:#bf8700,stroke:#845306,color:#fff
    style E fill:#1a7f37,stroke:#0f5323,color:#fff
    style F fill:#1a7f37,stroke:#0f5323,color:#fff
    style H fill:#cf222e,stroke:#82061e,color:#fff
```

The **strided convolution is the key trick**: it compresses 5511 input steps down to 1375 before the recurrent layers ever see the data. The GRUs then process a quarter of the sequence length, which is what makes training tractable — while the convolution simultaneously extracts local frequency patterns.

<details>
<summary><b>📋 Full layer-by-layer breakdown</b></summary>

<br/>

| # | Layer | Output shape | Params |
|--:|:--|:--|--:|
| 0 | `InputLayer` | (5511, 101) | 0 |
| 1 | `Conv1D` — 196 filters, kernel 15, stride 4 | (1375, 196) | 297,136 |
| 2 | `BatchNormalization` | (1375, 196) | 784 |
| 3 | `Activation` — ReLU | (1375, 196) | 0 |
| 4 | `Dropout` — 0.8 | (1375, 196) | 0 |
| 5 | `GRU` — 128 units | (1375, 128) | 125,184 |
| 6 | `Dropout` — 0.8 | (1375, 128) | 0 |
| 7 | `BatchNormalization` | (1375, 128) | 512 |
| 8 | `GRU` — 128 units | (1375, 128) | 99,072 |
| 9 | `Dropout` — 0.8 | (1375, 128) | 0 |
| 10 | `BatchNormalization` | (1375, 128) | 512 |
| 11 | `Dropout` — 0.8 | (1375, 128) | 0 |
| 12 | `TimeDistributed(Dense 1, sigmoid)` | (1375, 1) | 129 |

**Total 523,329** · Trainable 522,425 · Non-trainable 904 *(≈2.00 MB)*

</details>

---

## 📊 Results

<div align="center">

| | |
|:--|:--|
| **Training examples** | 801 *(synthesised in-notebook)* |
| **Loss** | Binary cross-entropy, per timestep |
| **Optimiser** | Adam · `lr=1e-6` · `β₁=0.9` · `β₂=0.999` |
| **Batch size** | 16 |
| **Epochs** | 500 *(single GPU)* |
| 🎯 **Dev set accuracy** | **83.0 %** |

</div>

---

## 🗂️ Repository structure

```text
trigger_word_detection_system/
├── trigger_word_detection_system.ipynb   ← the whole project, outputs included
├── requirements.txt
└── README.md

   ── not committed (DeepLearning.AI assets) ──
├── td_utils.py          helper module: graph_spectrogram, load_raw_audio, …
├── raw_data/
│   ├── activates/       ~1 s recordings of the word "activate"
│   ├── negatives/       ~1 s recordings of other words
│   └── backgrounds/     10 s ambient noise beds
├── audio_examples/      sample clips used by the notebook's figures
└── XY_dev/              X_dev.npy · Y_dev.npy — the held-out dev set
```

### Notebook walkthrough

| Cells | What happens |
|:--|:--|
| 0 – 1 | Imports, load raw audio, define `Tx=5511` · `n_freq=101` · `Ty=1375` |
| 2 – 8 | Data-synthesis helpers: random placement, overlap detection, label insertion |
| 9 | Generate the 801-example training set |
| 10 – 15 | Load the dev set, define and inspect the model |
| 16 – 18 | Compile, train for 500 epochs, evaluate |

---

## 🔧 Reproducing this

```bash
git clone https://github.com/mazder1/trigger_word_detection_system.git
cd trigger_word_detection_system

pip install -r requirements.txt
```

`pydub` also needs **FFmpeg** on your system path:

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt install ffmpeg
# Windows
winget install ffmpeg
```

Then drop `td_utils.py`, `raw_data/`, `audio_examples/` and `XY_dev/` into the project root — they ship with the Week 3 assignment workspace of **[Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)** on Coursera. Enrolled learners can download them straight from the notebook’s file browser.

```bash
jupyter notebook trigger_word_detection_system.ipynb
```

> 💡 A GPU is strongly recommended — the final cell trains for 500 epochs.

---

## 🧭 Notes & possible next steps

Things I’m aware of, kept here in the open rather than swept under the rug:

- **Learning rate of `1e-6` is very conservative** — roughly 100× below the usual choice for this model. It’s the main reason 500 epochs are needed; a higher rate would likely reach the same place far sooner.
- **Dropout at `0.8` appears four times**, which is aggressive regularisation for a 801-example set and may be capping the ceiling.
- **No inference path yet.** The notebook stops at evaluation. The natural next step is a `detect_triggerword()` that runs the model over a fresh clip and overlays a chime wherever the output crosses a threshold — plus saving the trained weights.
- **Accuracy alone flatters this task.** Since most timesteps are `0`, precision/recall on the positive window would tell a more honest story than 83 %.

---

## 🙏 Credits

This project follows the **Trigger Word Detection** assignment from the **[Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/)** by **Andrew Ng** and the team at **[DeepLearning.AI](https://www.deeplearning.ai/)** — specifically *Course 5: Sequence Models*, Week 3.

The problem framing, the synthesis strategy and the network architecture are theirs, and full credit goes to them for a genuinely brilliant piece of course design. The code in this notebook is my own implementation of it.

**All audio data and the `td_utils.py` helper module remain the property of DeepLearning.AI and are not redistributed in this repository.** If you want to run it, please obtain them through the course.

<div align="center">
<br/>
<sub>Educational project · shared for learning and portfolio purposes</sub>
</div>
