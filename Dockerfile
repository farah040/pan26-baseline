FROM webis/pan-pyterrier-baseline:dev-0.0.1

RUN pip3 install sentence-transformers faiss-cpu nltk
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

ADD e5_baseline.py /