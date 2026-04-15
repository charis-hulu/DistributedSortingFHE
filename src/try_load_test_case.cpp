#include <fstream>
#include <iostream>
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "key/key-ser.h"
using namespace lbcrypto;

int main() {
    std::string dir = "../test_cases/case_1";

    // ------------------- Load ciphertext chunk 0 -------------------
    std::string file0 = dir + "/ciphertext_chunk_0.txt";
    Ciphertext<DCRTPoly> c0;
    {
        std::ifstream ifs(file0, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open " << file0 << std::endl;
            return 1;
        }
        Serial::Deserialize(c0, ifs, SerType::BINARY);
    }

    // ------------------- Load ciphertext chunk 1 -------------------
    std::string file1 = dir + "/ciphertext_chunk_1.txt";
    Ciphertext<DCRTPoly> c1;
    {
        std::ifstream ifs(file1, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open " << file1 << std::endl;
            return 1;
        }
        Serial::Deserialize(c1, ifs, SerType::BINARY);
    }

    std::cout << "Ciphertexts loaded successfully." << std::endl;

    // ------------------- Get CryptoContext -------------------
    CryptoContext<DCRTPoly> cryptoContext = c0->GetCryptoContext();

    // ------------------- Load EvalMultKey -------------------
    std::string evalMultFile = dir + "/key-eval-mult.txt";
    std::ifstream ifsMult(evalMultFile, std::ios::binary);
    if (!ifsMult.is_open()) {
        std::cerr << "Failed to open EvalMultKey file: " << evalMultFile << std::endl;
        return 1;
    }
    cryptoContext->DeserializeEvalMultKey(ifsMult, SerType::BINARY);
    ifsMult.close();

    std::cout << "EvalMultKey loaded successfully." << std::endl;

    // ------------------- Load secret key -------------------
    KeyPair<DCRTPoly> keyPair;
    std::string secKeyFile = dir + "/key-secret.txt";
    std::ifstream ifsSec(secKeyFile, std::ios::binary);
    if (!ifsSec.is_open()) {
        std::cerr << "Failed to open secret key file: " << secKeyFile << std::endl;
        return 1;
    }
    Serial::Deserialize(keyPair.secretKey, ifsSec, SerType::BINARY);
    ifsSec.close();

    std::cout << "Secret key loaded successfully." << std::endl;

    // ------------------- Multiply -------------------
    Ciphertext<DCRTPoly> cMul = cryptoContext->EvalMult(c0, c1);
    std::cout << "Ciphertexts multiplied successfully." << std::endl;

    // ------------------- Decrypt -------------------
    Plaintext resultP;
    cryptoContext->Decrypt(keyPair.secretKey, cMul, &resultP);

    // Original chunk size
    size_t chunkSize = 2; // or whatever you used during encryption
    resultP->SetLength(chunkSize);
    
    std::vector<double> result = resultP->GetRealPackedValue();

    std::cout << "Decrypted multiplication result: " << result << std::endl;

    return 0;
}
