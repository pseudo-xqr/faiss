# cmake -B build . -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_CMMH_ALLOCATOR=ON -DCMAKE_BUILD_TYPE=Debug -G Ninja
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DFAISS_ENABLE_DPSK=ON -G Ninja
ninja -C build -j16 faiss
ninja -C build -j16 faiss_avx512_spr
ninja -C build install
# (cd build/faiss/python && python setup.py install)
