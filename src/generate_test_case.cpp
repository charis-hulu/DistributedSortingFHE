#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <nlohmann/json.hpp>

#include "utils-basics.h"
#include <openfhe.h>
#include "scheme/ckksrns/ckksrns-ser.h"
#include "key/key-ser.h"
#include <chrono>

using json = nlohmann::json;
using namespace lbcrypto;

#define LOG2(X) (size_t) std::ceil(std::log2((X)))



using Clock = std::chrono::high_resolution_clock;

double elapsed_seconds(const Clock::time_point& start,
                       const Clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

std::vector<int32_t> getRotationIndices
(
    const size_t matrixSize
)
{
    std::vector<int32_t> indices;

    int32_t index;
    for (size_t i = 0; i < LOG2(matrixSize); i++)
    {
        index = 1 << i;
        indices.push_back(index);   // sumColumns
        indices.push_back(-index);  // replicateColumn

        index = 1 << (LOG2(matrixSize) + i);
        indices.push_back(-index);  // replicateRow, sumRows

        index = (matrixSize * (matrixSize - 1) / (1 << (i + 1)));
        indices.push_back(index);   // transposeColumn
        indices.push_back(-index);  // transposeRow
    }

    return indices;
}

std::vector<std::vector<double>> splitVector(
    const std::vector<double>& vec,
    const size_t numSubvectors
)
{
    size_t subSize = vec.size() / numSubvectors;
    std::vector<std::vector<double>> result;
    auto it = vec.begin();
    for (size_t i = 0; i < numSubvectors; ++i)
    {
        result.push_back(std::vector<double>(it, it + subSize));
        it += subSize;
    }

    return result;
}

void print_memory_usage(const std::string& tag) {
    std::ifstream status("/proc/self/status");
    std::string line;
    long vmrss = 0;  // current RSS (KB)
    long vmhwm = 0;  // peak RSS (KB)

    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0)
            vmrss = std::stol(line.substr(6));
        else if (line.rfind("VmHWM:", 0) == 0)
            vmhwm = std::stol(line.substr(6));
    }

    std::cout << "[MEM] " << tag
              << " | RSS = " << vmrss / 1024.0 << " MB"
              << " | Peak = " << vmhwm / 1024.0 << " MB"
              << std::endl;
}

int main(int argc, char** argv) {
    
    print_memory_usage("start");
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <caseName>\n";
        return 1;
    }

    std::string caseName = argv[1];
    std::string dir = "/eagle/dist_relational_alg/chulu/fhe_test_cases/" + caseName;

    std::cout << "Using directory: " << dir << std::endl;

    std::string input_file = dir + "/input.json";

    // Read JSON file
    std::ifstream ifs(input_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: cannot open " << input_file << std::endl;
        return 1;
    }

    json config;
    ifs >> config;

    // Extract parameters from JSON
    const usint dg_c       = config["dg_c"];
    const usint df_c       = config["df_c"];
    const usint dg_i       = config["dg_i"];
    const usint df_i       = config["df_i"];
    // const usint indicatorDepth     = config["indicatorDepth"];
    const size_t subVectorLength   = config["subVectorLength"];
    std::vector<double> v          = config["v"].get<std::vector<double>>();
    const usint integralPrecision  = config["integralPrecision"];
    const usint decimalPrecision   = config["decimalPrecision"];
    const bool enableBootstrap     = config["enableBootstrap"];
    const usint ringDim            = config["ringDim"];
    const bool verbose             = config["verbose"];

    // Derived parameters
    const size_t vectorLength = v.size();
    const size_t numCiphertext = std::ceil((double) vectorLength / subVectorLength);
    const usint multiplicativeDepth = 4 * (dg_c + df_c + dg_i + df_i) + 6 + 3;
    const usint numSlots = subVectorLength * subVectorLength;

    // Print to verify
    std::cout << "dg_c = " << dg_c << std::endl;
    std::cout << "df_c = " << df_c << std::endl;
    std::cout << "dg_i = " << dg_i << std::endl;
    std::cout << "df_i = " << df_i << std::endl;
    // std::cout << "indicatorDepth = " << indicatorDepth << std::endl;
    std::cout << "subVectorLength = " << subVectorLength << std::endl;
    std::cout << "v = [ ";
    for (auto x : v) std::cout << x << " ";
    std::cout << "]" << std::endl;
    std::cout << "multiplicativeDepth = " << multiplicativeDepth << std::endl;
    std::cout << "numSlots = " << numSlots << std::endl;
    std::cout << "enableBootstrap = " << enableBootstrap << std::endl;
    std::cout << "verbose = " << verbose << std::endl;


    CryptoContext<DCRTPoly> cryptoContext = generateCryptoContext(
        integralPrecision,
        decimalPrecision,
        multiplicativeDepth,
        numSlots,
        enableBootstrap,
        ringDim,
        verbose
    );
    print_memory_usage("after generateCryptoContext");

    std::vector<int32_t> indices = getRotationIndices(subVectorLength);
    
    KeyPair<DCRTPoly> keyPair = keyGeneration(
        cryptoContext,
        indices,
        numSlots,
        enableBootstrap,
        verbose
    );
    print_memory_usage("after keyGeneration");
    
    // Save public key
    auto t0 = Clock::now();
    std::string pubKeyFile = dir + "/key-public.txt";
    std::ofstream ofsPub(pubKeyFile, std::ios::binary);
    Serial::Serialize(keyPair.publicKey, ofsPub, SerType::BINARY);
    ofsPub.close();
    std::cout << "Public key saved to " << pubKeyFile << std::endl;
    auto t1 = Clock::now();
    std::cout << "[TIME] Write public key: "
            << elapsed_seconds(t0, t1) << " s" << std::endl;

    print_memory_usage("after writing public key");

    // Save secret key
    t0 = Clock::now();
    std::string secKeyFile = dir + "/key-secret.txt";
    std::ofstream ofsSec(secKeyFile, std::ios::binary);
    Serial::Serialize( keyPair.secretKey, ofsSec, SerType::BINARY);
    ofsSec.close();
    std::cout << "Secret key saved to " << secKeyFile << std::endl;
    t1 = Clock::now();
    std::cout << "[TIME] Write secret key: "
            << elapsed_seconds(t0, t1) << " s" << std::endl;
    print_memory_usage("after writing secret key");

    std::vector<std::vector<double>> vTokens = splitVector(v, numCiphertext);
    
    std::vector<Ciphertext<DCRTPoly>> vC; 

    for (size_t i = 0; i < numCiphertext; i++) {
        t0 = Clock::now();
        std::cout << "Encrypt chunk " << i << std::endl;
        vC.push_back(cryptoContext->Encrypt(
            keyPair.publicKey,
            cryptoContext->MakeCKKSPackedPlaintext(vTokens[i])
        ));
        std::string cipherFile = dir + "/ciphertext_chunk_" + std::to_string(i) + ".txt";
        std::ofstream ofs_cipher(cipherFile, std::ios::out | std::ios::binary);

        Serial::Serialize(vC[i], ofs_cipher, SerType::BINARY);
        ofs_cipher.close();
        
        std::cout << "Ciphertext saved to " << cipherFile << std::endl;
        t1 = Clock::now();
        std::cout << "[TIME] Write chunk " << i << " : "
        << elapsed_seconds(t0, t1) << " s" << std::endl;
        print_memory_usage("after writing chunk");
    }
    

    std::string rotKeyFile = dir + "/key-eval-rot.txt";
    std::string multKeyFile = dir + "/key-eval-mult.txt";

    t0 = Clock::now();
    std::ofstream ofs_rot(rotKeyFile, std::ios::out | std::ios::binary);
    cryptoContext->SerializeEvalAutomorphismKey(ofs_rot, SerType::BINARY);
    
    ofs_rot.close();

    std::cout << "EvalAutomorphism key saved to " << dir + "/" + "key-eval-rot.txt" << std::endl;
    t1 = Clock::now();
    std::cout << "[TIME] Write EvalAutomorphism key: "
            << elapsed_seconds(t0, t1) << " s" << std::endl;
    print_memory_usage("after writing EvalAutomorphism key");
    

    t0 = Clock::now();
    std::ofstream ofs_mult(multKeyFile, std::ios::out | std::ios::binary);
    cryptoContext->SerializeEvalMultKey(ofs_mult, SerType::BINARY);
    ofs_mult.close();

    std::cout << "EvalMult key saved to " << multKeyFile << std::endl;
    t1 = Clock::now();
    std::cout << "[TIME] Write EvalMult key: "
            << elapsed_seconds(t0, t1) << " s" << std::endl;
    print_memory_usage("after writing EvalMult key");



    return 0;
}
