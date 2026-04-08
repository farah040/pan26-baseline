FROM webis/pan-pyterrier-baseline:dev-0.0.1

RUN pip3 install sentence-transformers faiss-cpu nltk
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"

ADD e5_baseline.py /