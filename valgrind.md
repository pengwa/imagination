Run Valgrind:

PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=yes --track-origins=yes --log-file="mem2.log" python orttraining/orttraining/test/python/bench.py
  389  cd ..
  390  wget https://sourceware.org/pub/valgrind/valgrind-3.17.0.tar.bz2
  391  bzip2 -d valgrind-3.17.0.tar.bz2
  392  tar -xf valgrind-3.17.0.tar
  393  ls
  394  cd valgrind-3.17.0
  395  ls
  396  ./config
  397  ./configure
  398  make
  399  make install
  400  sudo make
  401  sudo make install
  402  valgrind
  403  cd ..
  404  ls
  405  cd onnxruntime/
  406  PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=yes --track-origins=yes --log-file="mem3.log" python orttraining/orttraining/test/python/bench.py
