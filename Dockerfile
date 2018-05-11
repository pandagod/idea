FROM tensorflow/tensorflow
#Install python3
#Install scientific libraries
RUN pip install beautifulsoup4 langdetect spacy mysql-connector && python -m spacy download en
EXPOSE 8888