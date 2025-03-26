# GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Binary Text Classification

## Introduction
This repository is the official implementation of "GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Text Classification". In this work, we introduce GAProtoNet, a novel white-box Multi-head Graph Attention-based Prototypical Network designed to explain the decisions of text classification models built with LM encoders. In our approach, the input vector and prototypes are regarded as nodes within a graph, and we utilize multi-head graph attention to selectively construct edges between the input node and prototype nodes to learn an interpretable prototypical representation. Experiments on multiple public datasets show our approach achieves superior results without sacrificing the accuracy of the original black-box LMs. We also compare with four alternative prototypical network variations and our approach achieves the best accuracy and F1 among all. Our case study and visualization of prototype clusters also demonstrate the efficiency in explaining the decisions of black-box models built with LMs.

#### Paper: [GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Binary Text Classification](https://arxiv.org/abs/2409.13312)

## Requirements

- Python 3.8 or higher
- pip

## Installation

### Set up a virtual environment:
python3 -m venv venv
source venv/bin/activate

### create the environment
pip install -r requirements.txt

### (Optional) Install GPU support for PyTorch:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


## Download Datasets
1. Hotel Review: https://www.kaggle.com/datafiniti/ hotel-reviews
2. Yelp Polarity: https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews
3. IMDb: https://huggingface.co/datasets/stanfordnlp/imdb
4. Yahoo: https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset
5. Twitter: https://github.com/cardiffnlp/tweeteval

## Download Pretrained Models
1. Roberta: https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main
2. XLNet: https://huggingface.co/xlnet/xlnet-base-cased/tree/main
3. Distilbert: https://huggingface.co/distilbert/distilbert-base-uncased

## Set Parameters
1. Set dataset train/dev/test path in train.py --train_file_path --dev_file_path --test_file_path
2. Set pretrained model configs in train.py --bert_model_path --bert_config --llm_model_path --vocab_path --merge_path --tokenizer_path --tokenizer_config

### train the model
python src/train.py


## Citation
If you are interested in our work, feel free to cite:
```
@inproceedings{wen-etal-2025-gaprotonet,
    title = "{GAP}roto{N}et: A Multi-head Graph Attention-based Prototypical Network for Interpretable Text Classification",
    author = "Wen, Ximing and Tan, Wenjuan and Weber, Rosina",
    editor = "Rambow, Owen and Wanner, Leo and Apidianaki, Marianna and Al-Khalifa, Hend and Eugenio, Barbara Di and Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.661/",
    pages = "9891--9901",
    abstract = "Pretrained transformer-based Language Models (LMs) are well-known for their ability to achieve significant improvement on text classification tasks with their powerful word embeddings, but their \textit{black-box} nature, which leads to a lack of interpretability, has been a major concern. In this work, we introduce GAProtoNet, a novel \textit{white-box} Multi-head Graph Attention-based Prototypical Network designed to explain the decisions of text classification models built with LM encoders. In our approach, the input vector and prototypes are regarded as nodes within a graph, and we utilize multi-head graph attention to selectively construct edges between the input node and prototype nodes to learn an interpretable prototypical representation. During inference, the model makes decisions based on a linear combination of activated prototypes weighted by the attention score assigned for each prototype, allowing its choices to be transparently explained by the attention weights and the prototypes. Experiments on multiple public datasets show our approach achieves superior results without sacrificing the accuracy of the original black-box LMs. We also compare with four alternative prototypical network variations and our approach achieves the best accuracy and F1 among all. Our case study and visualization of prototype clusters also demonstrate the efficiency in explaining the decisions of black-box models built with LMs."
}
```

