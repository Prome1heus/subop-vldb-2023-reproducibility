#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace std;

const float miles_constant = 3959.0;
const float lat1 = 0.70984286;
const float lon1 = 1.2389197;

//const __m512 milesVec = _mm512_set1_ps(miles_constant);
//const __m512 lat1vec = _mm512_set1_ps(lat1);
//const __m512 lon1vec = _mm512_set1_ps(lon1);

float haversineCompiled(size_t size, float* __restrict__  lat2, float* __restrict__  lon2, float* __restrict__ result){
    auto range = tbb::blocked_range<size_t>(0, size);
    tbb::parallel_for(range, [&](const auto& r){
        float dlat, dlon, a, c, sinLat, sinLon;
        for (size_t i=r.begin(); i!=r.end(); i++){
            dlat = lat2[i] - lat1;
            dlon = lon2[i] - lon1;
            sinLat = sin(dlat / 2.0);
            sinLon = sin(dlon / 2.0);
            a = sinLat * sinLat + cos(lat1) * cos(lat2[i]) * sinLon * sinLon;
            c = 2.0 * asin(sqrt(a));
            result[i] = miles_constant * c;
        }
    });
    return result[0];
}

float haversineInterpreted(size_t size, vector<float>& lat2, vector<float>& lon2, vector<float>& result){
    auto range = tbb::blocked_range<size_t>(0, size / 64);
    tbb::parallel_for(range, [&](const auto& r){
        
    });
    return result[0];
}

int main(){
    size_t size;
    cin >> size;
    size = 1 << size;
    std::vector<float> lats(size, 0.0698132), lons(size, 0.0698132), result(size);

    auto compiledStart = std::chrono::high_resolution_clock::now();
    float res = haversineCompiled(size, lats.data(), lons.data(), result.data());
    auto compiledEnd = std::chrono::high_resolution_clock::now();
    auto compiledDuration = std::chrono::duration_cast<std::chrono::milliseconds>(compiledEnd - compiledStart);
    std::cout << "Compiled produced " << res << " in " << compiledDuration.count() << std::endl;

    return 0;
}
