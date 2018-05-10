FROM ubuntu
#Install python3
RUN apt-get update && apt-get intstall python3 && apt-get install python3-pip && apt-get build-dep python3-scipy
#Install scientific libraries
RUN pip3 install scikit-learn pandas beautifulsoup4 langdetect spacy && python -m spacy download en && pip3 install jupyter
RUN pip3 install --no-cache-dir --ignore-installed tensorflow==1.8