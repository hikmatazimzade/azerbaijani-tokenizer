# Azerbaijani Tokenizer

High-performance tokenizers built specifically for Azerbaijani language. 
Trained with BPE and Unigram algorithms. these tokenizers deliver 30% 
fewer tokens and 40% faster processing than multilingual alternatives.

## 🎬 Demo Video

https://github.com/user-attachments/assets/08017f10-575b-483c-8840-a6142927a00c

## 🎯 Live Demo

Try the Unigram tokenizer in real-time: [https://azetokenizer.vercel.app/](https://azetokenizer.vercel.app/)

## 📦 Hugging Face Models

- **Unigram Tokenizer**: [hikmatazimzade/azerbaijani-unigram-tokenizer](https://huggingface.co/hikmatazimzade/azerbaijani-unigram-tokenizer)
- **BPE Tokenizer**: [hikmatazimzade/azerbaijani-bpe-tokenizer](https://huggingface.co/hikmatazimzade/azerbaijani-bpe-tokenizer)

## 📊 Training Dataset

The tokenizers were trained on a large-scale Azerbaijani corpus created by merging and processing multiple datasets:
- **LocalDoc/AzTC**
- **allmalab/DOLLMA**

### Dataset Statistics
- **Total Words**: 1,380,222,892
- **Total Sentences**: 75,299,990

### Data Sources
The training corpus was compiled from various Azerbaijani text sources, filtered and cleaned to ensure high-quality language-specific content. The preprocessing pipeline included:
- Language detection (Azerbaijani vs English filtering)
- Cyrillic script removal
- Text normalization (Unicode normalization, whitespace handling)
- Sentence segmentation using NLTK with Turkish language model
- Filtering based on Azerbaijani-specific characters (ə, ı, ö, ü, ş, ğ, ç) and common words

## 🔧 Training Configuration

### BPE Tokenizer
- **Vocabulary Size**: 40,000 tokens
- **Training Samples**: 50,000,000 sentences
- **Character Coverage**: 0.9995
- **Max Sentence Length**: 4,192 bytes

### Unigram Tokenizer
- **Vocabulary Size**: 40,000 tokens
- **Training Samples**: 10,000,000 sentences
- **Character Coverage**: 0.9995
- **Max Sentence Length**: 4,192 bytes

## 📈 Performance Comparison

Benchmark results on 100 Azerbaijani sentences:

| Tokenizer | Token Count | Processing Time (seconds) | Reduction vs Baseline |
|-----------|------------|---------------------------|----------------------|
| **Azerbaijani Unigram** | **4,129** | **0.0078** | **31.5%** |
| **Azerbaijani BPE** | **4,189** | **0.0078** | **30.5%** |
| XLM-RoBERTa | 4,613 | 0.0129 | Baseline |
| BERT Multilingual | 6,030 | 0.0117 | -30.7% |

### Key Performance Insights
- **Token Efficiency**: Both custom tokenizers achieve ~30% token reduction compared to XLM-RoBERTa
- **Speed**: 40% faster processing time than multilingual alternatives
- **Unigram vs BPE**: Unigram slightly outperforms BPE in token count (1.4% fewer tokens)

## 💻 Installation

```bash
git clone https://github.com/yourusername/azerbaijani-tokenizer.git
cd azerbaijani-tokenizer

pip install uv
uv sync
```

## 🚀 Usage

```python
from transformers import AutoTokenizer

# Load from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("hikmatazimzade/azerbaijani-unigram-tokenizer")
# or
tokenizer = AutoTokenizer.from_pretrained("hikmatazimzade/azerbaijani-bpe-tokenizer")

# Tokenize
text = "Biz Bütöv Azərbaycançıyıq, bunu öz əməllərimizdə göstərmişik."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

# Decode
decoded_text = tokenizer.decode(token_ids)
```

## 📝 Example Tokenization

Input text:
```
"Uzun müddətdir ki, cəbhədə rəsmi olaraq atəşkəs hökm sürür."
```

Tokenization output (Unigram):
```
['▁Uzun', '▁müddətdir', '▁ki', ',', '▁cəbhədə', '▁rəsmi', '▁olaraq', '▁atəşkəs', '▁hökm', '▁sürür', '.']
```

## 🤝 Contributing

All contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache-2.0 license License - see the LICENSE file for details.

**Citation**

If you use these tokenizers in your research, please cite:

```bibtex
@misc{azerbaijani-tokenizer-2025,
  author = {Hikmat Azimzade},
  title = {Azerbaijani Language-Specific Tokenizers},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/hikmatazimzade}}
}
```