Run Valgrind:

PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=yes --track-origins=yes --log-file="mem2.log" python orttraining/orttraining/test/python/bench.py

cd ..
 
wget https://sourceware.org/pub/valgrind/valgrind-3.17.0.tar.bz2
 
bzip2 -d valgrind-3.17.0.tar.bz2
 
tar -xf valgrind-3.17.0.tar
 
ls

cd valgrind-3.17.0

ls

./config

./configure

make

make install

sudo make

sudo make install

valgrind

cd ..

ls

cd onnxruntime/

PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=yes --track-origins=yes --log-file="mem3.log" python orttraining/orttraining/test/python/bench.py
