#include <chrono>
#include <iostream>

auto t0 = std::chrono::high_resolution_clock::now();
...
auto t1 = std::chrono::high_resolution_clock::now();
std::chrono::duration<float> fs = t1 - t0;
// std::chrono::milliseconds d = std::chrono::duration_cast<ms>(fs);
std::cout << "PythonOp e2e latency(s): " << fs.count() << std::endl;
// std::cout << d.count() << "ms\n";
