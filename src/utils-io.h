#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "openfhe.h"

#include "scheme/ckksrns/ckksrns-ser.h"
#include "key/key-ser.h"

using namespace lbcrypto;

using json = nlohmann::json;

// JSON config struct
struct JsonConfig {
    size_t subVectorLength;
    size_t numCiphertext;
    size_t dg_c;
    size_t df_c;
    size_t dg_i;
    size_t df_i;
    std::vector<double> v;
};

JsonConfig readJsonConfig(const std::string& jsonFile);

Ciphertext<DCRTPoly> readChunk(const std::string& filePath);

KeyPair<DCRTPoly> readKey(const std::string& filePath);

void readEvalMultKey(CryptoContext<DCRTPoly> cc, const std::string& filePath);
void readEvalRotKey(CryptoContext<DCRTPoly> cc, const std::string& filePath);