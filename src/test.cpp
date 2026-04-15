// #include "utils-basics.h"
#include <mpi.h>
#include <iostream>
using namespace std;


void testSortingMultiCtxt(
    const size_t subVectorLength,
    const size_t numCiphertext,
    const usint compareDepth,
    const usint indicatorDepth,
    const bool tieCorrection = false
)
{

    const size_t vectorLength           = subVectorLength * numCiphertext;
    const usint integralPrecision       = 1;
    const usint decimalPrecision        = 59;
    const usint multiplicativeDepth     = compareDepth + indicatorDepth + 3 + (tieCorrection ? 3 : 0);
    const usint numSlots                = subVectorLength * subVectorLength;
    const bool enableBootstrap          = false;
    const usint ringDim                 = 0;
    const bool verbose                  = true;

    std::vector<int32_t> indices = getRotationIndices(subVectorLength);

    CryptoContext<DCRTPoly> cryptoContext = generateCryptoContext(
        integralPrecision,
        decimalPrecision,
        multiplicativeDepth,
        numSlots,
        enableBootstrap,
        ringDim,
        verbose
    );

    KeyPair<DCRTPoly> keyPair = keyGeneration(
        cryptoContext,
        indices,
        numSlots,
        enableBootstrap,
        verbose
    );

    std::vector<double> v = loadPoints1D(vectorLength);
    std::vector<std::vector<double>> vTokens = splitVector(v, numCiphertext);

    std::cout << "Vector: " << vTokens << std::endl;

    std::cout << "Expected sorting: " << sort(v) << std::endl;

    std::vector<Ciphertext<DCRTPoly>> vC(numCiphertext);
    for (size_t j = 0; j < numCiphertext; j++)
        vC[j] = cryptoContext->Encrypt(
            keyPair.publicKey,
            cryptoContext->MakeCKKSPackedPlaintext(vTokens[j])
        );

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Ciphertext<DCRTPoly>> resultC;
    if (tieCorrection)
        resultC = sortWithCorrection(
            vC,
            subVectorLength,
            -1.0, 1.0,
            depth2degree(compareDepth),
            depth2degree(indicatorDepth)
        );
    else
        resultC = sort(
            vC,
            subVectorLength,
            -1.0, 1.0,
            depth2degree(compareDepth),
            depth2degree(indicatorDepth)
        );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Runtime: " << elapsed_seconds.count() << "s" << std::endl;

    Plaintext resultP;
    std::vector<std::vector<double>> resultTokens(numCiphertext);
    for (size_t i = 0; i < numCiphertext; i++)
    {
        cryptoContext->Decrypt(keyPair.secretKey, resultC[i], &resultP);
        resultP->SetLength(subVectorLength * subVectorLength);
        std::vector<double> resultMatrix = resultP->GetRealPackedValue();
        std::vector<double> resultToken(subVectorLength);
        for (size_t j = 0; j < subVectorLength; j++)
            resultToken[j] = resultMatrix[j * subVectorLength];
        resultTokens[i] = resultToken;
    }
    std::vector<double> result = concatVectors(resultTokens);
    std::cout << "Sorting: " << result << std::endl;

}
// void distributeCryptoContext(int src, int nprocs) {

// }



int naiveDistributedRanking(int rank, int nprocs) {
    const size_t subVectorLength = 256;
    const size_t vectorLength = v.size();
    
    vector<vector<<double>> V_;

    vector<Ciphertext<DCRTPoly>> V;
    
    vector <double> v; 
    
    if(rank == 0) {
        
        v = loadPoints();
        
        V_ = split_vector();

        const usint integralPrecision       = 1;
        const usint decimalPrecision        = 42;
        const usint multiplicativeDepth     = compareDepth + indicatorDepth + 10;
        const usint numSlots                = vectorLength * vectorLength;
        const bool enableBootstrap          = false;
        const usint ringDim                 = 0;
        const bool verbose                  = true;

        
        cryptoContext = generateCryptoContext(
            integralPrecision,
            decimalPrecision,
            multiplicativeDepth,
            numSlots,
            enableBootstrap,
            ringDim,
            verbose
        );
        
        std::vector<int32_t> indices = getRotationIndices(vectorLength);
        KeyPair<DCRTPoly> keyPair = keyGeneration(
            cryptoContext,
            indices,
            numSlots,
            enableBootstrap,
            verbose
        );

        for(int i=0; i<numCipherText; i++) {
            V.push_back(Encrypt(V_[i]));
        }
    }

    distributeCryptoContext(0, cryptoContext); // MPI_Broadcast
    
    distributeVector(0, V); // MPI_Broadcast


    vector<vector<Ciphertext<DCRTPoly>> C;
    vector<Ciphertext<DCRTPoly>> R;


    #pragma omp parallel for
    for (size_t i = 0; i < numCipherText; i++) {
        vR[i] = ReplR(V[i]);
        vC[i] = ReplC(TransR(V[i]));
    }

    #pragma omp parallel for
    for(int j = 0; j < numCipherText; j++) {
        C[rank][j] = Compare(vR[rank], vC[j]);
    }


    for(int j=0; j < numCipherText; j++) {
        R[rank] += C[rank][j] 
    }

    R[rank] = SumC(R[rank]);
    

}

// int distributed_sorting() {
//     return 0;
// }


int main(int argc, char** argv) {
    int a;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "Hello from rank " << rank << " of " << size << endl;

    MPI_Finalize();
    return 0;
}