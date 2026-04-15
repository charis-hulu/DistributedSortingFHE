#include "utils-basics.h"
#include "utils-matrices.h"
#include "utils-communication.h"
#include "utils-ptxt.h"
#include "utils-eval.h"
#include <vector>

#include <mpi.h>
#include <iostream>
#include <openfhe.h>

#include "scheme/ckksrns/ckksrns-ser.h"
#include "key/key-ser.h"

#include <string>
#include <nlohmann/json.hpp>


#include <cmath>

#include <omp.h>

using namespace std;


using json = nlohmann::json;


struct JsonConfig {
    size_t subVectorLength;
    size_t numCiphertext;
    std::vector<int> v;
};

inline JsonConfig readJsonConfig(const std::string& jsonFile) {
    std::ifstream ifs(jsonFile);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonFile);
    }

    json j;
    ifs >> j;
    ifs.close();

    JsonConfig cfg;
    cfg.subVectorLength = j["subVectorLength"];
    cfg.v = j["v"].get<std::vector<int>>();
    cfg.numCiphertext = cfg.v.size() / cfg.subVectorLength;

    return cfg;
}

// Ciphertext<DCRTPoly> distributedRanking(
//     int rank,
//     const std::vector<Ciphertext<DCRTPoly>> &c,
//     const size_t subVectorLength,
//     const double leftBoundC,
//     const double rightBoundC,
//     const uint32_t degreeC,
//     KeyPair<DCRTPoly> keyPair
// ) {



//     return 0; 
// }


void printCiphertext(const std::string& mark, 
                     Ciphertext<DCRTPoly> c, 
                     const size_t matrixSize, 
                     int rank, 
                     KeyPair<DCRTPoly> keyPair) 
{
    CryptoContext<DCRTPoly> cryptoContext = c->GetCryptoContext();

    Plaintext resultP;
    cryptoContext->Decrypt(keyPair.secretKey, c, &resultP);
    resultP->SetLength(matrixSize);

    std::vector<double> resultMatrix = resultP->GetRealPackedValue();
    std::cout << "[" << mark << "] Rank " << rank << " ct = " << resultMatrix << std::endl;
}

int main(int argc, char** argv) {



    MPI_Init(&argc, &argv);

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }

    int rank;
    int nprocs;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);



    if(rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <case_x>\n";
                return 1;
        }
    }

    std::string caseName = argv[1];
    std::string dir = "./test_cases/" + caseName;
    std::string jsonFile = dir + "/input.json";
    
    JsonConfig cfg;
    try {
        cfg = readJsonConfig(jsonFile);
    } catch (std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Rank=" << rank 
              << " subVectorLength=" << cfg.subVectorLength
              << " numCiphertext=" << cfg.numCiphertext
              << std::endl;



    const usint compareDepth = 10;
    const usint indicatorDepth = 10;
    const size_t subVectorLength = cfg.subVectorLength;
    const size_t numCiphertext = cfg.numCiphertext;
    const size_t subMatrixSize = subVectorLength * subVectorLength;
    const size_t procWidth = static_cast<int>(std::sqrt(nprocs));
    const size_t compRegionWidth = numCiphertext / procWidth;
    cout << "compRegionWidth=" << compRegionWidth << std::endl;
    // const size_t numReqThreads = ctPerProc * ctPerProc;
    


    int dims[2] = {(int) procWidth , (int) procWidth};
    int periods[2] = {0, 0};

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);


    int row = coords[0];
    int col = coords[1];

    cout << "Rank=" << rank << " row=" << row << " col=" << col << endl; 

    double t_start, t_end;
    double t_load = 0.0, t_repl = 0.0, t_compare = 0.0, t_gather_rank = 0.0, t_sum_rank = 0.0, t_broadcast_rank = 0.0, t_submasking = 0.0, t_sum_sorting = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks
    t_start = MPI_Wtime();


    double t1 = MPI_Wtime();
    std::vector<Ciphertext<DCRTPoly>> ctInputs;

    for(size_t i=0; i < compRegionWidth; i++) {
        std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string((col * compRegionWidth) + i) + ".txt";
        Ciphertext<DCRTPoly> ct;
        {
            std::ifstream ifs(ctFile, std::ios::binary);
            if (!ifs.is_open()) {
                std::cerr << "Failed to open " << ctFile << std::endl;
                return 1;
            }
            Serial::Deserialize(ct, ifs, SerType::BINARY);
        }
        ctInputs.push_back(ct);
        std::cout << "Ciphertexts loaded successfully." << std::endl;      
    }

    if(row != col) {
        for(size_t i=0; i < compRegionWidth; i++) {
            std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string((row * compRegionWidth) + i) + ".txt";
            Ciphertext<DCRTPoly> ct;
            {
                std::ifstream ifs(ctFile, std::ios::binary);
                if (!ifs.is_open()) {
                    std::cerr << "Failed to open " << ctFile << std::endl;
                    return 1;
                }
                Serial::Deserialize(ct, ifs, SerType::BINARY);
            }
            ctInputs.push_back(ct);
            std::cout << "Ciphertexts loaded successfully." << std::endl;
        }
    }

    // ------------------- Get CryptoContext -------------------
    CryptoContext<DCRTPoly> cryptoContext = ctInputs[0]->GetCryptoContext();

    // KeyPair<DCRTPoly> keyPair;
    // std::string secKeyFile = dir + "/key-secret.txt";
    // std::ifstream ifsSec(secKeyFile, std::ios::binary);
    // if (!ifsSec.is_open()) {
    //     std::cerr << "Failed to open secret key file: " << secKeyFile << std::endl;
    //     return 1;
    // }
    // Serial::Deserialize(keyPair.secretKey, ifsSec, SerType::BINARY);
    // ifsSec.close();

    // std::cout << "Secret key loaded successfully." << std::endl;


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

    // ------------------- Load EvalRotKey -------------------
    std::string evalRotFile = dir + "/key-eval-rot.txt";
    std::ifstream ifsRot(evalRotFile, std::ios::binary);
    if (!ifsRot.is_open()) {
        std::cerr << "Failed to open EvalRotKey file: " << evalRotFile << std::endl;
        return 1;
    }
    cryptoContext->DeserializeEvalAutomorphismKey(ifsRot, SerType::BINARY);
    ifsRot.close();

    std::cout << "EvalRotKey loaded successfully." << std::endl;

    double t2 = MPI_Wtime();

    t_load = t2 - t1;
    

    
    double t3 = MPI_Wtime();

    std::vector<Ciphertext<DCRTPoly>> replR(compRegionWidth);
    std::vector<Ciphertext<DCRTPoly>> replC(compRegionWidth);

    if(row == col) {
        #pragma omp parallel for collapse(2)
        for(size_t loopID = 0; loopID < 2; loopID++) {
            for(size_t j=0; j < compRegionWidth; j++) {
                if(loopID == 0)
                {
                    replR[j] = replicateRow(ctInputs[j], subVectorLength);
                }
                else {
                    replC[j] = replicateColumn(transposeRow(ctInputs[j], subVectorLength, true), subVectorLength);
                }
            }
        }
    }
    else {
        #pragma omp parallel for collapse(2)
        for(size_t loopID = 0; loopID < 2; loopID++) {
            for(size_t j=0; j < compRegionWidth; j++) {
                if(loopID == 0) 
                {
                    replR[j] = replicateRow(ctInputs[j], subVectorLength);
                }
                else {
                    replC[j] = replicateColumn(transposeRow(ctInputs[compRegionWidth + j], subVectorLength, true), subVectorLength);
                }
            }
        }
    }

    double t4 = MPI_Wtime();
    t_repl = t4 - t3;
    // for(size_t i=0; i < compRegionWidth; i++) {
    //     printCiphertext("VR" + std::to_string(i), replR[i],  subVectorLength * subVectorLength, rank, keyPair);
    //     printCiphertext("VC" + std::to_string(i), replC[i],  subVectorLength * subVectorLength, rank, keyPair);
    
    // }

    double t5 = MPI_Wtime();

    std::vector<Ciphertext<DCRTPoly>> R(compRegionWidth);
    std::vector<bool> Rinitialized(compRegionWidth);
    Ciphertext<DCRTPoly> Cij;

    #pragma omp parallel for collapse(2)
    for(size_t i=0; i < compRegionWidth; i++) {
        for(size_t j=0; j < compRegionWidth; j++) {
            Cij = compare(
                replC[i], replR[j],
                -30, 20, depth2degree(compareDepth)
            );
            #pragma omp critical 
            {
                if(!Rinitialized[i]) {
                    R[i] = Cij;
                    Rinitialized[i] = true;
                }
                else {
                    R[i] = R[i] + Cij;
                }
            }
        }    
    }

    std::cout << "Finish comparison" << std::endl;
    double t6 = MPI_Wtime();
    t_compare = t6 - t5;
    //----------------------------
    // Split into row communicators
    //----------------------------
    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, row, col, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    
    double t7 = MPI_Wtime();
    std::vector<std::vector<Ciphertext<DCRTPoly>>> allRanks;
    Communication::gatherVectorToRoot(R, allRanks, 0, row_comm);
    double t8 = MPI_Wtime();
    t_gather_rank = t8 - t7;
    std::cout << "Finish communication" << std::endl;
    


    // #pragma omp parallel for
    // for(size_t i=0; i < row_size; i++) {
    //     for(size_t j=compRegionWidth + i; j < row_size; j+=compRegionWidth) {
    //         R[j] += allRanks[j];
    //     }
    // }

    double t9 = MPI_Wtime();
    if(row_rank == 0) {
        std::cout << "all rank size = " << allRanks.size() << std::endl;

        #pragma omp parallel for collapse(2)
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                #pragma omp critical
                R[i] += allRanks[j][i];
            }
        }
        
        R[0] = R[0] + 0.5;
        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
            R[i] = sumColumns(R[i], subVectorLength, true);
            R[i] = replicateColumn(R[i], subVectorLength);
        }
        // for(size_t i=0; i<compRegionWidth; i++) {
        //     printCiphertext("Rank", R[i], subVectorLength * subVectorLength, rank, keyPair);
        // }
        
    }
    double t10 = MPI_Wtime();
    t_sum_rank = t10 - t9;

    std::cout << "Finish summing up" << std::endl;
    

    double t11 = MPI_Wtime();
    Communication::broadcastVectorFromRoot<Ciphertext<DCRTPoly>>(R, 0, row_comm);
    double t12 = MPI_Wtime();
    t_broadcast_rank = t12 - t11;


    double t13 = MPI_Wtime();
    std::vector<std::vector<double>> subMasks(compRegionWidth, std::vector<double>(subVectorLength * subVectorLength));

    for(size_t i = 0; i < compRegionWidth; i++) {
        for(size_t j = 0; j < subVectorLength; j++) {
            for(size_t k = 0; k < subVectorLength; k++) {
                // We need to adjust this if we set rank 0 as client!
                subMasks[i][j * subVectorLength + k] = -static_cast<double>((col * subVectorLength * compRegionWidth) + k + 1);
            }
        }
    }

    for(size_t i = 0; i < compRegionWidth; i++) {
        std::cout << "Rank " << rank << " - subMask[" + std::to_string(i) + "] = " << subMasks[i] << std::endl;
    }


    std::vector<Ciphertext<DCRTPoly>> S(compRegionWidth);
    std::vector<bool> Sinitialized(compRegionWidth);

    #pragma omp parallel for collapse(2)
    for(size_t i=0; i < compRegionWidth; i++) {
        for(size_t j=0; j < compRegionWidth; j++) {

            Cij = indicator(
                    R[i] + subMasks[j],
                    -0.5, 0.5,
                    -1.01 * subMatrixSize, 1.01 * subMatrixSize,
                    depth2degree(indicatorDepth)
            ) * replC[i];

            #pragma omp critical 
            {
                if(!Sinitialized[i]) {
                    S[i] = Cij;
                    Sinitialized[i] = true;
                }
                else {
                    S[i] = S[i] + Cij;
                }
            }
        }    
    }
    double t14 = MPI_Wtime();
    t_submasking = t14 - t13;



    double t15 = MPI_Wtime();
    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, col, row, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    Communication::gatherVectorToRoot(S, allRanks, 0, col_comm);

    if(col_rank == 0) {
        std::cout << "all rank size = " << allRanks.size() << std::endl;

        #pragma omp parallel for collapse(2)
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                #pragma omp critical
                S[i] += allRanks[j][i];
            }
        }
        
        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
            S[i] = sumRows(S[i], subVectorLength, true);
        }
        std::vector<Ciphertext<DCRTPoly>> R(compRegionWidth);
        
        // for(size_t i=0; i<compRegionWidth; i++) {
        //     printCiphertext("S", S[i], subVectorLength * subVectorLength, rank, keyPair);
        // }
        
    }

    double t16 = MPI_Wtime();
    t_sum_sorting = t16-t15;


    t_end = MPI_Wtime();
    double t_total = t_end - t_start; 
    

    std::vector<double> local_times = {t_load, t_repl, t_compare, t_gather_rank, t_sum_rank, t_broadcast_rank, t_submasking, t_sum_sorting, t_total};
    std::vector<double> agg_times(local_times.size(), 0.0);


    MPI_Reduce(local_times.data(), agg_times.data(), local_times.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::ofstream csvFile("timing.csv");
        csvFile << "nranks,nthreads,num_ciphertext,"
                "load_time,replicate_time,compare_time,"
                "gather_rank_time,sum_rank_time,broadcast_rank_time,"
                "submask_time,sum_sort_time,total_time\n";

        csvFile << nprocs << "," << nthreads << "," << numCiphertext;
        for(double t : agg_times) {
            csvFile << "," << t;
        }
        csvFile << "\n";
        csvFile.close();
        std::cout << "Aggregate timing CSV saved to timing_aggregate.csv\n";
    }

    MPI_Finalize();



    return 0;
}