# 🧠 Brain Stroke Prediction using Deep Neural Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system that predicts the likelihood of brain stroke in patients using Deep Neural Networks, achieving **92.57% test accuracy** with a **0.97 stroke-class recall** after addressing class imbalance with SMOTE. The project includes a complete ML pipeline, web application, and detailed analysis.

## 🎯 Project Overview

This project addresses a critical healthcare challenge by developing an AI-powered system to predict stroke risk in patients. Stroke is the second leading cause of death globally, and early prediction can significantly improve patient outcomes through preventive care.

### Key Achievements
- **92.57% Test Accuracy** with **0.97 stroke-class recall** (post-SMOTE), up from a pre-SMOTE baseline recall of just 0.07
- **IEEE Best Technical Presentation** (500+ audience)
- **Government of India Copyright** awarded
- **Co-authored IEEE Publication**: "Stroke Risk Prediction Using Deep Neural Networks: Empowering Healthcare Services for Early Identification and Prevention," 2023 IEEE NMITCON — [IEEE Xplore #10276249](https://ieeexplore.ieee.org/document/10276249)
- **Production-ready Streamlit web application**

## 🚀 Features

### Machine Learning Pipeline
- **Data Preprocessing**: Handles categorical variables, missing values, and feature scaling
- **SMOTE Implementation**: Addresses class imbalance in medical datasets
- **Deep Neural Network**: 3-layer architecture with ReLU activation
- **Model Optimization**: Adam optimizer with binary cross-entropy loss
- **Comprehensive Evaluation**: Confusion matrix, classification report, and accuracy metrics

### Web Application
- **Interactive Interface**: User-friendly Streamlit dashboard
- **Real-time Prediction**: Instant stroke risk assessment
- **Input Validation**: Comprehensive form validation for medical parameters
- **Visual Feedback**: Clear risk indicators and confidence levels

### Data Analysis
- **Exploratory Data Analysis**: Comprehensive dataset exploration
- **Correlation Analysis**: Feature importance and relationships
- **Visualization**: Matplotlib and Seaborn plots for insights
- **Statistical Analysis**: Detailed performance metrics and validation

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning utilities
- **Pandas/NumPy** - Data manipulation and analysis
- **Streamlit** - Web application framework

### Data Processing
- **SMOTE** - Synthetic Minority Oversampling Technique
- **StandardScaler** - Feature normalization
- **LabelEncoder** - Categorical variable encoding
- **Train-Test Split** - Model validation strategy

### Visualization & Analysis
- **Matplotlib** - Statistical plotting
- **Seaborn** - Advanced data visualization
- **Jupyter Notebook** - Interactive development environment

## 📊 Dataset & Features

### Input Features
- **Demographics**: Age, Gender, Residence type
- **Medical History**: Hypertension, Heart disease, Ever married
- **Lifestyle**: Work type, Smoking status
- **Biometrics**: Average glucose level, BMI

### Dataset Statistics
- **Total Samples**: 4,981 patients
- **Features**: 10 input variables
- **Target**: Binary classification (Stroke/No Stroke)
- **Class Distribution**: Handled with SMOTE oversampling

## 🏗️ Project Structure

```
Brain-Stroke-Prediction-Using-DNN/
├── app_py.py                 # Streamlit web application
├── Final.ipynb              # Complete ML pipeline notebook
├── model_weights.h5         # Trained model weights
├── README.md               # Project documentation
└── requirements.txt        # Dependencies (to be added)
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install streamlit tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/shettyrohit0810/Brain-Stroke-Prediction-using-DNN.git
   cd Brain-Stroke-Prediction-using-DNN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   streamlit run app_py.py
   ```

4. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Enter patient details in the form
   - Get instant stroke risk prediction

### Jupyter Notebook
```bash
jupyter notebook Final.ipynb
```

## 📈 Model Performance

### Before vs. After SMOTE

| Metric | Pre-SMOTE (baseline) | Post-SMOTE (current) |
|---|---|---|
| Stroke-Class Recall | **0.07** | **0.97** |
| Test Accuracy | ~92% | **92.57%** |

The pre-SMOTE model's ~92% accuracy was almost entirely a majority-class artifact — it correctly identified only 7% of actual stroke cases (recall of 0.07). Applying SMOTE to correct the class imbalance raised stroke-class recall to 0.97 while holding overall test accuracy at 92.57%, making the model far more useful for its actual purpose: catching at-risk patients rather than just predicting the majority class.

### Model Architecture
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)             (None, 64)                704       
dense_4 (Dense)             (None, 32)                2080      
dense_5 (Dense)             (None, 1)                 33        
=================================================================
Total params: 2,817
Trainable params: 2,817
```

## 🔬 Technical Deep Dive

### Data Preprocessing Pipeline
1. **Categorical Encoding**: Label encoding for categorical variables
2. **Feature Scaling**: StandardScaler for numerical features
3. **SMOTE Application**: Synthetic oversampling for class balance
4. **Train-Validation Split**: 80-20 split with random state

### Model Training Process
1. **Architecture Design**: 3-layer DNN with ReLU activation
2. **Optimization**: Adam optimizer with binary cross-entropy
3. **Training**: 200 epochs with early stopping
4. **Validation**: 20% validation split for monitoring

### Web Application Features
- **Input Validation**: Range checks and data type validation
- **Preprocessing**: Real-time data transformation
- **Prediction**: Model inference with confidence scores
- **User Interface**: Intuitive form design with clear feedback

## 🎯 Use Cases

### Healthcare Applications
- **Clinical Decision Support**: Assist doctors in stroke risk assessment
- **Preventive Care**: Identify high-risk patients for early intervention
- **Population Health**: Screen large populations for stroke risk
- **Research**: Support medical research and clinical studies

### Educational Applications
- **ML Learning**: Demonstrate deep learning concepts
- **Healthcare AI**: Showcase AI applications in medicine
- **Data Science**: Example of end-to-end ML pipeline

## 🏆 Recognition & Impact

- **IEEE Best Technical Presentation** - Presented to 500+ audience
- **Government of India Copyright** - Official recognition for innovation
- **Co-authored IEEE Publication** - "Stroke Risk Prediction Using Deep Neural Networks: Empowering Healthcare Services for Early Identification and Prevention," 2023 IEEE International Conference on Network, Multimedia and Information Technology (NMITCON), [IEEE Xplore #10276249](https://ieeexplore.ieee.org/document/10276249)
- **Real-world Application** - Production-ready healthcare solution

## 🔮 Future Enhancements

### Technical Improvements
- **Model Optimization**: Hyperparameter tuning and architecture search
- **Feature Engineering**: Additional medical features and biomarkers
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Real-time Data**: Integration with hospital systems

### Application Features
- **Mobile App**: Cross-platform mobile application
- **API Development**: RESTful API for third-party integration
- **Dashboard**: Advanced analytics and reporting
- **Multi-language**: Support for multiple languages

## 👨‍💻 Developer Information

**Rohit Ramesh Shetty**  
*MS CS Student at USC | AI/ML & Full Stack Enthusiast*

- **Portfolio**: [rohitshetty.netlify.app](https://rohitshetty.netlify.app)
- **LinkedIn**: [in/shettyrohit0810](https://linkedin.com/in/shettyrohit0810)
- **Email**: rshetty@usc.edu
- **GitHub**: [@shettyrohit0810](https://github.com/shettyrohit0810)

### Professional Experience
- **AI/ML Intern** at Delphi, BTB, and Spay India
- **Full Stack Developer** with expertise in React, Angular, Node.js
- **Cloud Platforms**: GCP, AWS, Azure (certified)
- **Technologies**: Python, Java, Kotlin, JavaScript, TypeScript

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: rshetty@usc.edu
- **LinkedIn**: [Rohit Shetty](https://linkedin.com/in/shettyrohit0810)
- **GitHub**: [@shettyrohit0810](https://github.com/shettyrohit0810)

---

⭐ **Star this repository** if you found it helpful!

🔗 **Connect with me** for AI/ML and software engineering opportunities!

---

*This project demonstrates advanced machine learning techniques applied to healthcare, showcasing both technical expertise and real-world impact potential.*