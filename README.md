# 🎤 Voice Gender and Age Prediction

A machine learning project built by **Rahul Goyal** to predict a speaker's **gender** and **age group** from voice recordings using **Random Forest classifiers**. The model processes `.wav` files to extract audio features, then classifies the speaker's demographic profile.

---

## 📌 Features

- Predicts **Gender**: Male / Female  
- Predicts **Age Group**: (e.g., Child / Adult / Senior)  
- Uses audio feature extraction with `librosa`  
- Built with **Random Forest classifiers**  
- CLI script to run predictions on new `.wav` files  
- Modular structure with utils and model folders  

---

## 📁 Project Structure

```
voice_gender_age_prediction_project/
│
├── 📁 data/                     # Sample audio & CSVs (full data via Drive)
│   ├── gender.csv
│   ├── age.csv
│   └── sample_audio/
│
├── 📁 models/                   # Trained Random Forest models
│   ├── gender_model.pkl
│   └── age_model.pkl
│
├── 📁 utils/                    # Feature extraction logic
│   └── feature_extraction.py
│
├── generate_features.py        # Extract features from audio files
├── predict.py                  # Predict gender and age from a .wav file
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🧠 How It Works

1. Load audio using `librosa`
2. Extract features like MFCCs, chroma, and mel spectrograms
3. Use trained Random Forest models to predict:
   - Gender
   - Age group
4. Display the result in terminal/console

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rahulgoyal9000/voice-gender-age-prediction.git
cd voice-gender-age-prediction
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

Due to GitHub's file size limitations, the full dataset (audio + labels) is hosted externally:

🔗 [Download Full Dataset](https://drive.google.com/drive/folders/1cQS10z7kmSbmT_5xkfUH_HaK-RsgpgII?usp=sharing)

After downloading, extract into the `data/` folder like this:

```
data/
├── gender.csv
├── age.csv
└── all_audio_files/
```

> This repo includes **sample files** for quick testing.

---

## 🚀 Usage

### A. Generate Features from Audio

```bash
python generate_features.py
```

This creates intermediate CSV files with features extracted from `.wav` audio.

---

### B. Predict Gender and Age from a `.wav` File

```bash
python predict.py --file data/sample_audio/test1.wav
```

#### ✅ Example Output:
```
Predicted Gender: Male
Predicted Age Group: Adult
```

---

## ⚙️ Model Used

- **Random Forest Classifier** from Scikit-learn
- One model each for:
  - Gender prediction
  - Age group prediction

---

## 📉 Limitations

- Trained on public datasets with mostly **Western accents**
- Performance may vary on **Indian-accented voices**
- Background noise can impact accuracy
- Age group classification is broad and not exact

---

## 🚀 Future Improvements

- Add more data with Indian/regional accents  
- Improve models with Deep Learning (e.g., LSTM or CNN)  
- Add emotion or language prediction  

---

## 📬 Author

Made with ❤️ by **Rahul Goyal**  
✉️ 2023CSB1150@iitrpr.ac.in

---

## 📜 License

This project is licensed under the [MIT License](LICENSE)


