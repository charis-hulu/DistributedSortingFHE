#include "utils-io.h"
#include <fstream>
#include <iostream>


// Read a dummy ciphertext from a plain text file
// Ciphertext<DCRTPoly> readChunk(const std::string& filePath) {
//     std::ifstream ifs(filePath);
//     Ciphertext<DCRTPoly> ct;
//     {
//         std::ifstream ifs(filePath, std::ios::binary);
//         if (!ifs.is_open()) {
//             std::cerr << "Failed to open " << filePath << std::endl;
//             return 1;
//         }
//         Serial::Deserialize(ct, ifs, SerType::BINARY);
//     }
//     return ct;
// }

Ciphertext<DCRTPoly> readChunk(const std::string& filePath) {
    std::ifstream ifs(filePath, std::ios::binary);

    if (!ifs.is_open()) {
        std::cerr << "Failed to open " << filePath << std::endl;
        return Ciphertext<DCRTPoly>();   // return empty
    }

    Ciphertext<DCRTPoly> ct;
    Serial::Deserialize(ct, ifs, SerType::BINARY);

    return ct;
}

// KeyPair<DCRTPoly> readKey(const std::string& filePath) {
//     std::ifstream ifs(filePath, std::ios::binary);

//     if (!ifs.is_open()) {
//         std::cerr << "Failed to open secret key file: " << filePath << std::endl;
//         return KeyPair<DCRTPoly>();   // empty key
//     }
//     KeyPair<DCRTPoly> keyPair;
//     Serial::Deserialize(keyPair, ifs, SerType::BINARY);
//     return keyPair;
// }

// Read JSON config
JsonConfig readJsonConfig(const std::string& jsonFile) {
    std::ifstream ifs(jsonFile);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonFile);
    }

    json j;
    ifs >> j;
    ifs.close();

    JsonConfig cfg;
    cfg.subVectorLength = j["subVectorLength"];
    cfg.v = j["v"].get<std::vector<double>>();
    cfg.numCiphertext = cfg.v.size() / cfg.subVectorLength;
    cfg.dg_c = j["dg_c"];
    cfg.df_c = j["df_c"];
    cfg.dg_i = j["dg_i"];
    cfg.df_i = j["df_i"];
    return cfg;
}

KeyPair<DCRTPoly> readKey(const std::string& filePath) {
    std::ifstream ifs(filePath, std::ios::binary);

    if (!ifs.is_open()) {
        std::cerr << "Failed to open secret key file: " << filePath << std::endl;
        return KeyPair<DCRTPoly>();   // empty key
    }
    KeyPair<DCRTPoly> keyPair;
    Serial::Deserialize(keyPair.secretKey, ifs, SerType::BINARY);
    return keyPair;
}



void readEvalMultKey(CryptoContext<DCRTPoly> cc, const std::string& filePath) {
    std::ifstream ifs(filePath, std::ios::binary);

    if (!ifs.is_open()) {
        std::cerr << "Failed to open EvalMultKey file: " << filePath << std::endl;
        return;
    }

    cc->DeserializeEvalMultKey(ifs, SerType::BINARY);
}

void readEvalRotKey(CryptoContext<DCRTPoly> cc, const std::string& filePath) {
    std::ifstream ifs(filePath, std::ios::binary);

    if (!ifs.is_open()) {
        std::cerr << "Failed to open EvalRotKey file: " << filePath << std::endl;
        return;
    }

    cc->DeserializeEvalAutomorphismKey(ifs, SerType::BINARY);
}
