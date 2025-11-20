# TaBERT Model RAG Q&A on Tabular Data Pipeline

This repository implements a Retrieval-Augmented Generation (RAG) question-answering system for tabular data using the TaBERT model. TaBERT is a pre-trained language model that learns joint representations of natural language utterances and structured tables, making it ideal for semantic parsing and table understanding tasks.

## Overview

TaBERT (Tabular BERT) is pre-trained on a massive corpus of 26M Web tables and their associated natural language context. This project extends TaBERT's capabilities to build a RAG pipeline that can answer questions about tabular data by combining retrieval and generation techniques.

## Features

- Pre-trained TaBERT models for understanding natural language and table structures
- RAG pipeline for question-answering on tabular data
- Support for both base and large model variants
- Integration with modern transformer architectures
- Efficient table encoding and context representation

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for training)
- Conda package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd tabert-rag-qa
   ```

2. **Create and activate the conda environment**
   ```bash
   bash scripts/setup_env.sh
   conda activate tabert
   ```

3. **Install the package and dependencies**
   ```bash
   pip install --editable .
   pip install -r requirements.txt
   ```

### Library Compatibility Note

The pre-trained models are compatible with both `pytorch-pretrained-bert` (legacy) and the latest `transformers` library. The conda environment installs both versions by default. You can uninstall `pytorch-pretrained-bert` if you prefer using only the latest `transformers` library.

## Pre-trained Models

Pre-trained TaBERT models are available for download. You can download them using the following commands:

```bash
pip install gdown

# TaBERT_Base_(k=1)
gdown 'https://drive.google.com/uc?id=1-pdtksj9RzC4yEqdrJQaZu4-dIEXZbM9'

# TaBERT_Base_(K=3)
gdown 'https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj'

# TaBERT_Large_(k=1)
gdown 'https://drive.google.com/uc?id=1eLJFUWnrJRo6QpROYWKXlbSOjRDDZ3yZ'

# TaBERT_Large_(K=3)
gdown 'https://drive.google.com/uc?id=17NTNIqxqYexAzaH_TgEfK42-KmjIRC-g'
```

**Note:** Please uncompress the tarball files before usage.

## Quick Start

### Loading a Pre-trained Model

```python
from table_bert import TableBertModel

# Load from checkpoint
model = TableBertModel.from_pretrained(
    'path/to/pretrained/model/checkpoint.bin',
)
```

### Encoding Tables and Context

```python
from table_bert import Table, Column

# Create a table
table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)

# Define context
context = 'show me countries ranked by GDP'

# Encode context and table
context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)

# Output shapes
# context_encoding.shape -> torch.Size([1, 7, 768])
# column_encoding.shape -> torch.Size([1, 2, 768])
```

### Using Vanilla BERT

You can also initialize a TaBERT model from standard BERT parameters:

```python
from table_bert import TableBertModel

model = TableBertModel.from_pretrained('bert-base-uncased')
```

## Project Structure

```
├── examples/              # Example applications
├── preprocess/           # Data preprocessing scripts
├── scripts/              # Setup and utility scripts
├── table_bert/           # Main TaBERT implementation
├── utils/                # Training data generation utilities
├── train.py              # Model training script
└── requirements.txt      # Python dependencies
```

## Example Applications

TaBERT can be used as a general-purpose representation learning layer for semantic parsing tasks over database tables. Example applications are available in the `examples` folder.

## Advanced Usage

### Training Data Generation

Generate training data for masked language modeling:

```bash
output_dir=data/train_data/vanilla_tabert
mkdir -p ${output_dir}

python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus data/preprocessed_data/tables.jsonl \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 15 \
    --max_context_len 128 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column|type|value' \
    --column_delimiter "[SEP]"
```

### Model Training

Train a vanilla TaBERT model:

```bash
mkdir -p data/runs/vanilla_tabert

python train.py \
    --task vanilla \
    --data-dir data/train_data/vanilla_tabert \
    --output-dir data/runs/vanilla_tabert \
    --table-bert-extra-config '{}' \
    --train-batch-size 8 \
    --gradient-accumulation-steps 32 \
    --learning-rate 2e-5 \
    --max-epoch 10 \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128
```

## Data Processing

For information on extracting and preprocessing table corpora from CommonCrawl and Wikipedia, please refer to the detailed documentation in the `preprocess` directory.

## Citation

If you use TaBERT in your research, please cite the original paper:

```bibtex
@inproceedings{yin20acl,
    title = {Ta{BERT}: Pretraining for Joint Understanding of Textual and Tabular Data},
    author = {Pengcheng Yin and Graham Neubig and Wen-tau Yih and Sebastian Riedel},
    booktitle = {Annual Conference of the Association for Computational Linguistics (ACL)},
    month = {July},
    year = {2020}
}
```

## License

This project is licensed under CC-BY-NC 4.0.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This project builds upon the original TaBERT implementation and extends it for RAG-based question-answering on tabular data.