# MedAi

# How to run
### Steps:

Clone the repository

```bash
Project repo: https://github.com/nikhilgangrade1605/MedAi.git
```
### Step1- Create a conda env 

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### Step2- Install the requirements

```bash
pip install -r requirements.txt
```

### Index PDF documents

Place searchable PDF files in `data/` and run:

```bash
python store_index.py
```

The script indexes all PDFs in `data/` into the Pinecone `medai` index. PDFs
without extractable text are skipped, and stable chunk IDs make reruns update
existing records instead of adding duplicates. The retrieval notebook connects
to this same index using the configured sentence-transformer embeddings.
