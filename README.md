# Real-Reveal: A Deepfake Detection Framework

Real-Reveal is a deepfake detection framework that uses Convolutional Neural Networks (CNNs) to classify images as real or fake. It compares the performance of three distinct loss functions—Binary Cross-Entropy Loss, Focal Loss, and Hinge Loss—to provide insights into optimal configurations for deepfake detection models.

---

## Project Highlights

### Purpose
With deepfakes becoming increasingly prevalent, detecting manipulated content is crucial to counter misinformation, privacy violations, and security threats. Real-Reveal offers a robust framework to identify deepfake images and provides a comparative analysis of loss functions for model optimization.

### Key Features
- **Three Loss Functions Compared:**
  - Binary Cross-Entropy (BCE): A reliable default for binary classification.
  - Focal Loss: Effective for imbalanced datasets.
  - Hinge Loss: Useful for margin-based classification.
- **Dataset:** Over 190,000 images from Kaggle’s deepfake and real image datasets.
- **Performance Metrics:** Accuracy, precision, recall, F1-score, and loss.
- **Pre-trained Models:** Available for download on [Hugging Face](https://huggingface.co/anjaliisharmaa).

---

## Getting Started

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Real-Reveal.git
   cd Real-Reveal
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset
The dataset used in this project includes 190,335 images of real and manipulated faces. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images). Place it in the `data/` directory and ensure the structure matches the preprocessing scripts.

### Pre-trained Models
Download the pre-trained models from Hugging Face:
- [Binary Cross-Entropy Model](https://huggingface.co/yourusername/BCE_Model)
- [Focal Loss Model](https://huggingface.co/yourusername/Focal_Model)
- [Hinge Loss Model](https://huggingface.co/yourusername/Hinge_Model)

Place the models in the `models/` directory.

### Running the Models
Run the Jupyter notebooks in the `notebooks/` directory to:
1. Train models.
2. Evaluate performance metrics.
3. Visualize results.

Example:
```bash
jupyter notebook notebooks/Training_BCE.ipynb
```

---

## Results

### Performance Metrics
| Loss Function     | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------------------|-------------------|---------------------|---------------|-----------------|
| Binary Cross Entropy | 91.28%          | 85.28%             | 0.2087        | 0.3401          |
| Focal Loss         | 92.70%           | 85.94%             | 0.0118        | 0.0234          |
| Hinge Loss         | 91.58%           | 85.90%             | 0.5837        | 0.6446          |

### Key Observations
- **Binary Cross-Entropy Loss**: Balanced performance with moderate overfitting.
- **Focal Loss**: Best for imbalanced datasets, achieving the highest training accuracy.
- **Hinge Loss**: Shows potential but requires further tuning for improved generalization.

### Visualizations
- **Accuracy and Loss Curves:** Available in the `results/` folder.
- **Sample Predictions:** Visual outputs of the models on test data are included in the `results/` folder.

---

## Research Paper
For an in-depth analysis, refer to the [Research Paper](./Research_Paper.pdf).

---

## Future Work
- Implementing hybrid models using a combination of loss functions.
- Expanding the dataset to include more diverse and challenging deepfakes.
- Real-time detection capabilities for videos and audios.

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments
Special thanks to:
- [Kaggle](https://www.kaggle.com) for the dataset.
- [Hugging Face](https://huggingface.co) for hosting the models.
- Indira Gandhi Delhi Technical University for Women for research support.

---

## Contributing
Contributions are welcome! Please raise an issue or submit a pull request for any suggestions or improvements.

---

### Contact
For queries, contact **Anjali Sharma** at [anjali018btcseai23@igdtuw.ac.in](mailto:anjali018btcseai23@igdtuw.ac.in).

