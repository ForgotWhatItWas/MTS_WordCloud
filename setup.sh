# setup.sh
!sudo apt-get install swig3.0
!wget http://files.deeppavlov.ai/embeddings/ft_native_300_ru_twitter_nltk_word_tokenize.bin
!wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
!wget https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz
!wget https://github.com/bakwc/JamSpell-models/raw/master/ru.tar.gz
!tar -xvf en.tar.gz
!tar -xvf ru.tar.gz