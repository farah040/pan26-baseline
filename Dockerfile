FROM webis/pan-pyterrier-baseline:dev-0.0.1

RUN pip3 install sentence-transformers faiss-cpu nltk

ADD baseline.py /
ADD e5_baseline.py /