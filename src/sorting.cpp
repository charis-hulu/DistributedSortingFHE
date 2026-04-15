#include "sorting.h"
#include "utils-communication.h"
#include "utils-basics.h"
#include <omp.h>


void printDetailedMemoryUsage(int rank, const std::string& tag) {
    std::ifstream file("/proc/self/status");
    std::string line;

    std::string vmrss, vmhwm;

    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0)
            vmrss = line;
        if (line.rfind("VmHWM:", 0) == 0)
            vmhwm = line;
    }

    std::cerr << "[Rank " << rank << "] "
              << tag << "\n  "
              << vmrss << "\n  "
              << vmhwm << std::endl;
}

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

std::vector<Ciphertext<DCRTPoly>> distributedSortingGridWise(    
    std::string caseName
) 
{

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string dir = "test_cases/" + caseName;
    std::string jsonFile = dir + "/input.json";
    

    JsonConfig cfg = readJsonConfig(dir + "/input.json");
    
    const usint compareDepth = 10;
    const usint indicatorDepth = 10;


    const size_t subVectorLength = cfg.subVectorLength;
    const size_t subMatrixSize = subVectorLength * subVectorLength;
    const size_t numCiphertext = cfg.numCiphertext;
    int procWidth = static_cast<int>(std::sqrt(nprocs));
    const size_t compRegionWidth = numCiphertext / procWidth;
    
    std::cout << "Rank=" << rank 
              << " subVectorLength=" << cfg.subVectorLength
              << " numCiphertext=" << cfg.numCiphertext
              << " compRegionWidth=" << compRegionWidth
              << std::endl;


    int dims[2] = {procWidth, procWidth};
    int periods[2] = {0, 0};

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    

    int row = coords[0];
    int col = coords[1];

    std::vector<Ciphertext<DCRTPoly>> ctInputs;
    
    MPI_Barrier(cart_comm);

    double t_read_start = MPI_Wtime();


    for (size_t i = 0; i < compRegionWidth; i++) {
        size_t index = (col * compRegionWidth) + i;
        std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string(index) + ".txt";
        ctInputs.push_back(readChunk(ctFile));
        std::cout << "Ciphertext chunk " << index << " loaded successfully." << std::endl;
    }

    if (row != col) {
        for (size_t i = 0; i < compRegionWidth; i++) {
            size_t index = (row * compRegionWidth) + i;
            std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string(index) + ".txt";
            ctInputs.push_back(readChunk(ctFile));
            std::cout << "Ciphertext chunk " << index << " loaded successfully." << std::endl;
        }
    }


    CryptoContext<DCRTPoly> cryptoContext = ctInputs[0]->GetCryptoContext();

    std::string evalMultFile = dir + "/key-eval-mult.txt";
    readEvalMultKey(cryptoContext, evalMultFile);

    std::string evalRotFile = dir + "/key-eval-rot.txt";
    readEvalRotKey(cryptoContext, evalRotFile);

    
    std::string secKeyFile = dir + "/key-secret.txt";
    KeyPair<DCRTPoly> keyPair = readKey(secKeyFile);

    double t_read_end = MPI_Wtime();

    MPI_Barrier(cart_comm);

    // PHASE 2: Replication

    double t_repl_start = MPI_Wtime();

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

    double t_repl_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish replication" << std::endl;
    
    // for(size_t i=0; i < compRegionWidth; i++) {
    //     std::cout << "VR_" << i << " Rank " << rank << "ct = " << replR[i] << std::endl;
    //     std::cout << "VC_" << i << " Rank " << rank << "ct = " << replC[i] << std::endl;
    // }
    
    MPI_Barrier(cart_comm);

    // PHASE 3: COMPARE

    double t_compare_start = MPI_Wtime();


    std::vector<Ciphertext<DCRTPoly>> R(compRegionWidth);
    std::vector<bool> Rinitialized(compRegionWidth);
    Ciphertext<DCRTPoly> Cij;
    
    #pragma omp parallel for collapse(2)
    for(size_t i=0; i < compRegionWidth; i++) {
        for(size_t j=0; j < compRegionWidth; j++) {
            Cij = compare(
                replC[i], replR[j], -30, 20, depth2degree(compareDepth)
            );
            // std::cout << "[C_" << i << j << "] Rank " << rank << " ct = " << Cij << std::endl;
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
    
    double t_compare_end = MPI_Wtime();
    std::cout << "Rank " << rank << "- finish compare" << std::endl;



    //----------------------------
    // Split into row communicators
    //----------------------------
    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, row, col, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    MPI_Barrier(cart_comm);

    // PHASE 4: Gather comparison results to root
    double t_gather_rank_start = MPI_Wtime();
    std::vector<std::vector<Ciphertext<DCRTPoly>>> allRanks;
    Communication::gatherVectorToRoot(R, allRanks, 0, row_comm);

    double t_gather_rank_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather rank" << std::endl;
    MPI_Barrier(cart_comm);


    // PHASE 5: Sum to compute rank on each row
    double t_sum_rank_start = MPI_Wtime();
    
    if(row_rank == 0) {
        
        // for(int i=0; i<compRegionWidth; i++) {
        //     for(size_t j=1; j < (size_t) row_size; j++) {
        //         std::cout << "[AllRanks_" << i << j << "] Rank " << rank << " ct = " << allRanks[i][j] << std::endl;
        //     }
            
        // }
        
        // std::cout << "all rank size = " << allRanks.size() << std::endl;

        #pragma omp parallel for
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                R[i] = R[i] + allRanks[j][i];
            }
        }
        

        // R[0] = R[0] + 0.5;

        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
            // R[i] = R[i] + 0.5;
            R[i] = sumColumns(R[i], subVectorLength, true);
            R[i] = replicateColumn(R[i], subVectorLength);
            R[i] = R[i] + 0.5;
            // std::cout << "[R_" << i << "] Rank " << rank << " ct = " << R[i] << std::endl;
        }
        
        
        // printCiphertext("Rank", R[row_rank], subVectorLength * subVectorLength, rank, keyPair);
    }
    double t_sum_rank_end = MPI_Wtime();


    MPI_Barrier(cart_comm);

    // PHASE 6: Broadcast ranking results
    double t_broadcast_rank_start = MPI_Wtime();

    Communication::broadcastVectorFromRoot<Ciphertext<DCRTPoly>>(R, 0, row_comm);

    double t_broadcast_rank_end = MPI_Wtime();

    MPI_Barrier(cart_comm);

    // PHASE 7: Indicator

    double t_indicator_start = MPI_Wtime();


    std::vector<std::vector<double>> subMasks(compRegionWidth, std::vector<double>(subVectorLength * subVectorLength));

    for(size_t i = 0; i < compRegionWidth; i++) {
        for(size_t j = 0; j < subVectorLength; j++) {
            for(size_t k = 0; k < subVectorLength; k++) {
                // We need to adjust this if we set rank 0 as client!
                subMasks[i][j * subVectorLength + k] = -static_cast<double>((col * subVectorLength * compRegionWidth) + (i * subVectorLength) + k + 1);
            }
        }
    }

    // for(size_t i = 0; i < compRegionWidth; i++) {
    //     std::cout << "[subMasks_" << i  << "] Rank " << rank << " v = " << subMasks[i] << std::endl;
    // }
    
    std::vector<Ciphertext<DCRTPoly>> S(compRegionWidth);
    std::vector<bool> Sinitialized(compRegionWidth);

    #pragma omp parallel for collapse(2)
    for(size_t j=0; j < compRegionWidth; j++) {
        for(size_t i=0; i < compRegionWidth; i++) {
            // Cij = indicatorAdv(
            //     R[i] + subMask[i],
            //     vectorLength,
            //     dg_i, df_i
            // );
            Cij = indicator(
                R[i] + subMasks[j],             
                -0.5, 0.5,
                -1.01 * subMatrixSize, 1.01 * subMatrixSize,
            depth2degree(indicatorDepth)
            ) * replC[i];
            
            // std::cout << "[C_" << i << j << "] Rank " << rank << " ct = " << Cij << std::endl;
            #pragma omp critical 
            {
                if(!Sinitialized[j]) {
                    S[j] = Cij;
                    Sinitialized[j] = true;
                }
                else {
                    S[j] = S[j] + Cij;
                }
            }
        }
        // std::cout << "[ReplC_" << i  << "] Rank " << rank << " ct = " << replC[i] << std::endl;    
    }

    double t_indicator_end = MPI_Wtime();

    MPI_Barrier(cart_comm);


    // Phase 8: Gather sort

    double t_gather_sort_start = MPI_Wtime();


    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, col, row, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    Communication::gatherVectorToRoot(S, allRanks, 0, col_comm);
    
    double t_gather_sort_end = MPI_Wtime();

    MPI_Barrier(cart_comm);


    // Phase 9: Sum sort
    double t_sum_sort_start = MPI_Wtime();

    if(col_rank == 0) {
        // std::cout << "all rank size = " << allRanks.size() << std::endl;
        
        #pragma omp parallel for
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                #pragma omp critical
                S[i] = S[i] + allRanks[j][i];
            }
        }
        
        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
            S[i] = sumRows(S[i], subVectorLength, true);
        }

        // for (size_t i = 0; i < compRegionWidth; i++) {
        //     std::cout << "S[" << i << "] Rank " << rank << " ct = " << S[i] << std::endl;
        // }
        // std::vector<Ciphertext> R(compRegionWidth);
        printCiphertext("S", S[col_rank], subVectorLength, rank, keyPair);
    }
    double t_sum_sort_end = MPI_Wtime();


    double t_read = t_read_end - t_read_start;
    double t_repl = t_repl_end - t_repl_start;
    double t_compare = t_compare_end - t_compare_start;
    double t_gather_rank = t_gather_rank_end - t_gather_rank_start;
    double t_sum_rank = t_sum_rank_end - t_sum_rank_start;
    double t_broadcast_rank = t_broadcast_rank_end - t_broadcast_rank_start;
    double t_indicator = t_indicator_end - t_indicator_start;
    double t_gather_sort = t_gather_sort_end - t_gather_sort_start;
    double t_sum_sort = t_sum_sort_end - t_sum_sort_start;

    double max_read, max_repl, max_compare, max_gather_rank, max_sum_rank;
    double max_broadcast_rank, max_indicator, max_gather_sort, max_sum_sort;

    // ---------------------------------------------------------
    // Perform MAX reduction for each timing phase
    // ---------------------------------------------------------
    MPI_Reduce(&t_read,           &max_read,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_repl,           &max_repl,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compare,        &max_compare,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather_rank,    &max_gather_rank,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_rank,       &max_sum_rank,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_broadcast_rank, &max_broadcast_rank, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_indicator,      &max_indicator,      1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather_sort,    &max_gather_sort,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_sort,       &max_sum_sort,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     // int threads_per_rank = 0;
    //     // #pragma omp parallel
    //     // {
    //     //     #pragma omp single
    //     //     threads_per_rank = omp_get_num_threads();
    //     // }

    //     // const char *filename = "timing.csv";

    //     // FILE *fp = fopen(filename, "a+");   // create if not exist, append if exist
    //     // if (!fp) {
    //     //     perror("fopen");

    //     // }

    //     // // Check if file is empty
    //     // fseek(fp, 0, SEEK_END);
    //     // long size = ftell(fp);

    //     // if (size == 0) {
    //     //     // Write header only once
    //     //     fprintf(fp,
    //     //         "nranks,threads_per_rank,read,repl,compare,gather_offset,sum_offset,"
    //     //         "gather_rank,sum_rank,broadcast_rank,"
    //     //         "indicator,gather_sort,sum_sort\n");
    //     // }

    //     // // Append one row
    //     // fprintf(fp,
    //     //     "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
    //     //     nprocs,
    //     //     threads_per_rank,
    //     //     max_read,
    //     //     max_repl,
    //     //     max_compare,
    //     //     max_gather_offset,
    //     //     max_sum_offset,
    //     //     max_gather_rank,
    //     //     max_sum_rank,
    //     //     max_broadcast_rank,
    //     //     max_indicator,
    //     //     max_gather_sort,
    //     //     max_sum_sort
    //     // );

    //     // fclose(fp);
    // }

    return S;

}

std::vector<Ciphertext<DCRTPoly>> distributedSortingWithCorrectionGridWise(    
    std::string caseName
)
{

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string dir = "/eagle/dist_relational_alg/chulu/fhe_test_cases/" + caseName;
    std::string jsonFile = dir + "/input.json";
    

    JsonConfig cfg = readJsonConfig(dir + "/input.json");
    
    // const usint compareDepth = 10;
    // const usint indicatorDepth = 10;
    const usint dg_c       = cfg.dg_c;
    const usint df_c       = cfg.df_c;
    const usint dg_i       = cfg.dg_i;
    const usint df_i       = cfg.df_i;
    
    const size_t subVectorLength = cfg.subVectorLength;
    // const size_t subMatrixSize = subVectorLength * subVectorLength;
    const size_t numCiphertext = cfg.numCiphertext;
    int procWidth = static_cast<int>(std::sqrt(nprocs));
    const size_t compRegionWidth = numCiphertext / procWidth;
    const size_t vectorLength = numCiphertext * subVectorLength;
    
    std::cout << "Rank=" << rank 
              << " subVectorLength=" << cfg.subVectorLength
              << " numCiphertext=" << cfg.numCiphertext
              << " compRegionWidth=" << compRegionWidth
              << std::endl;


    int dims[2] = {procWidth, procWidth};
    int periods[2] = {0, 0};

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    

    int row = coords[0];
    int col = coords[1];

    
    MPI_Barrier(cart_comm);

    double t_read_start = MPI_Wtime();

    std::vector<Ciphertext<DCRTPoly>> ctInputs;

    
    // PHASE 1: IO READING


    for (size_t i = 0; i < compRegionWidth; i++) {
        size_t index = (col * compRegionWidth) + i;
        std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string(index) + ".txt";
        ctInputs.push_back(readChunk(ctFile));
        std::cout << "Ciphertext chunk " << index << " loaded successfully." << std::endl;
    }

    if (row != col) {
        for (size_t i = 0; i < compRegionWidth; i++) {
            size_t index = (row * compRegionWidth) + i;
            std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string(index) + ".txt";
            ctInputs.push_back(readChunk(ctFile));
            std::cout << "Ciphertext chunk " << index << " loaded successfully." << std::endl;
        }
    }
    // print_memory_usage("after loading ciphertext");

    CryptoContext<DCRTPoly> cryptoContext = ctInputs[0]->GetCryptoContext();

    // std::cout << "Rank " << rank << " - Crypto address 0 : " << ctInputs[0]->GetCryptoContext().get() << std::endl;

    // std::cout << "Rank " << rank << " - Crypto address 1 : " << ctInputs[1]->GetCryptoContext().get() << std::endl;

    std::string evalMultFile = dir + "/key-eval-mult.txt";
    readEvalMultKey(cryptoContext, evalMultFile);
    std::cout << "Rank = " << rank << " - key eval multiplication loaded successfully" << std::endl;
    // print_memory_usage("after loading key multiplication");

    std::string evalRotFile = dir + "/key-eval-rot.txt";
    readEvalRotKey(cryptoContext, evalRotFile);
    // print_memory_usage("after loading key rotation");
    std::cout << "Rank = " << rank << " - key eval rotation loaded successfully" << std::endl;

    std::string secKeyFile = dir + "/key-secret.txt";
    KeyPair<DCRTPoly> keyPair = readKey(secKeyFile);
    std::cout << "Rank = " << rank << " - secret key loaded successfully" << std::endl;

    double t_read_end = MPI_Wtime();

    MPI_Barrier(cart_comm);

    // PHASE 2: Replication

    double t_repl_start = MPI_Wtime();

    std::vector<Ciphertext<DCRTPoly>> replR(compRegionWidth);
    std::vector<Ciphertext<DCRTPoly>> replC(compRegionWidth);
    
    if(row == col) {
        #pragma omp parallel for collapse(2)
        for(size_t loopID = 0; loopID < 2; loopID++) {
            for(size_t j=0; j < compRegionWidth; j++) {
                if(loopID == 0)
                {
                    #pragma omp critical
                    {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}  
                    replR[j] = replicateRow(ctInputs[j], subVectorLength);
                    #pragma omp critical
                    {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                } else {
                    #pragma omp critical
                    {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}
                    replC[j] = replicateColumn(transposeRow(ctInputs[j], subVectorLength, true), subVectorLength);
                    #pragma omp critical
                    {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
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
                    #pragma omp critical
                    {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}  
                    replR[j] = replicateRow(ctInputs[j], subVectorLength);
                    #pragma omp critical
                    {std::cout << "ReplicateRow - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                }
                else {
                    #pragma omp critical
                    {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - START" << std::endl;}
                    replC[j] = replicateColumn(transposeRow(ctInputs[compRegionWidth + j], subVectorLength, true), subVectorLength);
                    #pragma omp critical
                    {std::cout << "ReplicateColumn - j = " << j << " - thread_id = " << omp_get_thread_num() << " - END" << std::endl;}
                }
            }
        }
    }

    // for(size_t i=0; i < compRegionWidth; i++) {
    //     printCiphertext("VR_" + std::to_string(i), replR[i], subVectorLength, rank, keyPair);
    // }

    // for(size_t i=0; i < compRegionWidth; i++) {
    //     printCiphertext("VC_" + std::to_string(i), replC[i], subVectorLength, rank, keyPair);
    // }

    if(rank == 0) {
            std::cout << "Rank " << rank << " replR levels: " << replR[0]->GetLevel() << std::endl;
            std::cout << "Rank " << rank << " replC levels: " << replC[0]->GetLevel() << std::endl;
    }


    double t_repl_end = MPI_Wtime();
    std::cout << "Rank " << rank << "- finish replication" << std::endl;


    MPI_Barrier(cart_comm);

    // PHASE 3: COMPARE
    double t_compare_start = MPI_Wtime();
    

    std::vector<Ciphertext<DCRTPoly>> R(compRegionWidth);
    std::vector<bool> Rinitialized(compRegionWidth);
    

    std::vector<Ciphertext<DCRTPoly>> Ev(compRegionWidth);
    std::vector<bool> Evinitialized(compRegionWidth);


    std::vector<double> triangleMask(subVectorLength * subVectorLength, 1.0);
    for(size_t i=0; i<subVectorLength; i++) {
        for(size_t j=0; j<subVectorLength; j++) {
            if(i < j) {
                triangleMask[i * subVectorLength + j] = 0.0;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for(size_t i=0; i < compRegionWidth; i++) {
        for(size_t j=0; j < compRegionWidth; j++) {
        Ciphertext<DCRTPoly> Cij = compareAdv(
                                    replC[i], replR[j],
                                    dg_c, df_c
                                );
            // Ciphertext<DCRTPoly> Cij = compare(
            //     replC[i], replR[j], -1, 1, depth2degree(compareDepth)
            // );

            // std::cout << "[C_" << i << j << "] Rank " << rank << " ct = " << Cij << std::endl;

            Ciphertext<DCRTPoly> Eij = 4 * (1 - Cij) * Cij;
        

            Ciphertext<DCRTPoly> Pij = Eij;
            
            if(row < col) {
                Pij = Eij * 0;
            }
            else if(row == col) {
                if(i < j) {
                    Pij = Eij * 0;
                }
                else if(i == j) {
                    Pij = Eij * triangleMask;
                }
            }
            
            Eij = Eij * 0.5;
            
            Eij = Pij - Eij;
            // std::cout << "[E_" << i << j <<"] Rank " << rank << " ct = " << Eij << std::endl;

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
            
            #pragma omp critical
            {
                if(!Evinitialized[i]) {
                    Ev[i] = Eij;
                    Evinitialized[i] = true;
                }
                else {
                    Ev[i] = Ev[i] + Eij;
                }
            }
        }    
    }

    if(rank == 0) {
            std::cout << "Rank " << rank << " R levels: " << R[0]->GetLevel() << std::endl;
            std::cout << "Rank " << rank << " Ev levels: " << Ev[0]->GetLevel() << std::endl;
    }

    double t_compare_end = MPI_Wtime();
    std::cout << "Rank " << rank << "- finish comparison" << std::endl;


    MPI_Barrier(cart_comm);


    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, row, col, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    // PHASE 4: Gather offset results to root
    double t_gather_offset_start = MPI_Wtime();

    std::vector<std::vector<Ciphertext<DCRTPoly>>> allEv;
     
    Communication::gatherVectorToRoot(Ev, allEv, 0, row_comm);

    double t_gather_offset_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather offset" << std::endl;

    MPI_Barrier(cart_comm);


    // PHASE 5: Sum offset results to root
    double t_sum_offset_start = MPI_Wtime();

    if(row_rank == 0) {
        // #pragma omp parallel for collapse(2)
        #pragma omp parallel for
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                // #pragma omp critical
                Ev[i] = Ev[i] + allEv[j][i];
                
            }        
        }
        
        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
                
            Ev[i] = sumColumns(Ev[i], subVectorLength, true);

            // std::cout << "[Ev_" << i << "] Rank " << rank << " ct = " << Ev[i] << std::endl;
            Ev[i] = replicateColumn(transposeRow(Ev[i], subVectorLength, true), subVectorLength);
            Ev[i] = Ev[i] - 0.5;
            
            // std::cout << "[R_" << i << "] Rank " << rank << " ct = " << R[i] << std::endl;
        }
    }

    double t_sum_offset_end = MPI_Wtime();

    if(rank == 0) {
            std::cout << "Rank " << rank << " R_after levels: " << R[0]->GetLevel() << std::endl;
            std::cout << "Rank " << rank << " Ev_after levels: " << Ev[0]->GetLevel() << std::endl;
    }

    std::cout << "Rank " << rank << "- finish sum offset" << std::endl;



    MPI_Barrier(cart_comm);

    // PHASE 6: Gather rank to root
    double t_gather_rank_start = MPI_Wtime();

    std::vector<std::vector<Ciphertext<DCRTPoly>>> allRanks;

    Communication::gatherVectorToRoot(R, allRanks, 0, row_comm);

    double t_gather_rank_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather rank" << std::endl;

    MPI_Barrier(cart_comm);

    // PHASE 7: Sum to compute rank on each row

    double t_sum_rank_start = MPI_Wtime();
    if(row_rank == 0) {
        
        // #pragma omp parallel for collapse(2)
        #pragma omp parallel for
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                // #pragma omp critical
                R[i] = R[i] + allRanks[j][i];
            }
        }
        

        // R[0] = R[0] + 0.5;

        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
            // R[i] = R[i] + 0.5;
            R[i] = sumColumns(R[i], subVectorLength, true);
            R[i] = replicateColumn(R[i], subVectorLength);
            R[i] = R[i] + 0.5;

            R[i] = R[i] + Ev[i];
            // std::cout << "[R_" << i << "] Rank " << rank << " ct = " << R[i] << std::endl;
        }
        
        // for(size_t i=0; i < compRegionWidth; i++) {
        //     printCiphertext("R", R[i], subVectorLength, rank, keyPair);
        // }
        
        if(rank == 0) {
                std::cout << "Rank " << rank << " R_final levels: " << R[0]->GetLevel() << std::endl;
        }
    }

    double t_sum_rank_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish sum rank" << std::endl;

    MPI_Barrier(cart_comm);

    // PHASE 8: Broadcast ranking results
    double t_broadcast_rank_start = MPI_Wtime();
    Communication::broadcastVectorFromRoot<Ciphertext<DCRTPoly>>(R, 0, row_comm);
    double t_broadcast_rank_end = MPI_Wtime();

    MPI_Barrier(cart_comm);

    // PHASE 9: Indicator

    double t_indicator_start = MPI_Wtime();
    std::vector<std::vector<double>> subMasks(compRegionWidth, std::vector<double>(subVectorLength * subVectorLength));

    for(size_t i = 0; i < compRegionWidth; i++) {
        for(size_t j = 0; j < subVectorLength; j++) {
            for(size_t k = 0; k < subVectorLength; k++) {
                // We need to adjust this if we set rank 0 as client!
                subMasks[i][j * subVectorLength + k] = -static_cast<double>((col * subVectorLength * compRegionWidth) + (i * subVectorLength) + k + 1);
            }
        }
    }


    std::vector<Ciphertext<DCRTPoly>> S(compRegionWidth);
    std::vector<bool> Sinitialized(compRegionWidth);

    #pragma omp parallel for collapse(2)
    for(size_t j=0; j < compRegionWidth; j++) {
        for(size_t i=0; i < compRegionWidth; i++) {
            Ciphertext<DCRTPoly> Cij = indicatorAdv(
                R[i] + subMasks[j],
                vectorLength,
                dg_i, df_i
            ) * replC[i];

            #pragma omp critical 
            {
                if(!Sinitialized[j]) {
                    S[j] = Cij;
                    Sinitialized[j] = true;
                }
                else {
                    S[j] = S[j] + Cij;
                }
            }
        }
        // std::cout << "[ReplC_" << i  << "] Rank " << rank << " ct = " << replC[i] << std::endl;    
    }

    // for(size_t i=0; i < compRegionWidth; i++) {
    //     printCiphertext("S_before_" + std::to_string(i), S[i], subVectorLength, rank, keyPair);
    // }

    if(rank == 0) {
            std::cout << "Rank " << rank << " S levels: " << S[0]->GetLevel() << std::endl;
    }

    double t_indicator_end = MPI_Wtime();
    std::cout << "Rank " << rank << "- finish indicator" << std::endl;

    MPI_Barrier(cart_comm);

    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, col, row, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);


    // PHASE 10: Gather sort
    double t_gather_sort_start = MPI_Wtime();
    Communication::gatherVectorToRoot(S, allRanks, 0, col_comm);
    double t_gather_sort_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather sort" << std::endl;

    MPI_Barrier(cart_comm);


    // PHASE 11: Sum sort
    double t_sum_sort_start = MPI_Wtime();
    if(col_rank == 0) {
        // std::cout << "all rank size = " << allRanks.size() << std::endl;

        #pragma omp parallel for
        for(size_t i=0; i<compRegionWidth; i++) {
            for(size_t j=1; j< (size_t) row_size; j++) {
                #pragma omp critical
                S[i] = S[i] + allRanks[j][i];
            }
        }
        
        #pragma omp parallel for
        for(size_t i=0; i < compRegionWidth; i++) {
            S[i] = sumRows(S[i], subVectorLength);
        }

        // for(size_t i=0; i < compRegionWidth; i++) {
        //     printCiphertext("S_" + std::to_string(i) , S[i], subVectorLength, rank, keyPair);
        // }
    }

    double t_sum_sort_end = MPI_Wtime();

    if(rank == 0) {
            std::cout << "Rank " << rank << " S_final levels: " << S[0]->GetLevel() << std::endl;
    }

    std::cout << "Rank " << rank << "- finish sum sort indicator" << std::endl;

    double t_read = t_read_end - t_read_start;
    double t_repl = t_repl_end - t_repl_start;
    double t_compare = t_compare_end - t_compare_start;
    double t_gather_rank = t_gather_rank_end - t_gather_rank_start;

    double t_gather_offset = t_gather_offset_end - t_gather_offset_start;
    double t_sum_offset = t_sum_offset_end - t_sum_offset_start;

    double t_sum_rank = t_sum_rank_end - t_sum_rank_start;
    double t_broadcast_rank = t_broadcast_rank_end - t_broadcast_rank_start;
    double t_indicator = t_indicator_end - t_indicator_start;
    double t_gather_sort = t_gather_sort_end - t_gather_sort_start;
    double t_sum_sort = t_sum_sort_end - t_sum_sort_start;


    double max_read, max_repl, max_compare, max_gather_rank, max_sum_rank, max_gather_offset, max_sum_offset;
    double max_broadcast_rank, max_indicator, max_gather_sort, max_sum_sort;


    // ---------------------------------------------------------
    // Perform MAX reduction for each timing phase
    // ---------------------------------------------------------
    MPI_Reduce(&t_read,           &max_read,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_repl,           &max_repl,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compare,        &max_compare,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t_gather_offset,    &max_gather_offset,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_offset,       &max_sum_offset,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t_gather_rank,    &max_gather_rank,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_rank,       &max_sum_rank,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_broadcast_rank, &max_broadcast_rank, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_indicator,      &max_indicator,      1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather_sort,    &max_gather_sort,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_sort,       &max_sum_sort,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        
        printf("\n=== Aggregated Max Timing (seconds) ===\n");
        printf("read           : %f\n", max_read);
        printf("repl           : %f\n", max_repl);
        printf("compare        : %f\n", max_compare);
        printf("gather_offset  : %f\n", max_gather_offset);
        printf("sum_offset     : %f\n", max_sum_offset);
        printf("gather_rank    : %f\n", max_gather_rank);
        printf("sum_rank       : %f\n", max_sum_rank);
        printf("broadcast_rank : %f\n", max_broadcast_rank);
        printf("indicator      : %f\n", max_indicator);
        printf("gather_sort    : %f\n", max_gather_sort);
        printf("sum_sort       : %f\n", max_sum_sort);
        printf("=======================================\n");

        int threads_per_rank = 0;
        #pragma omp parallel
        {
            #pragma omp single
            threads_per_rank = omp_get_num_threads();
        }

        const char *filename = "timing.csv";

        FILE *fp = fopen(filename, "a+");   // create if not exist, append if exist
        if (!fp) {
            perror("fopen");

        }

        // Check if file is empty
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);

        if (size == 0) {
            // Write header only once
            fprintf(fp,
                "nranks,threads_per_rank,read,repl,compare,gather_offset,sum_offset,"
                "gather_rank,sum_rank,broadcast_rank,"
                "indicator,gather_sort,sum_sort\n");
        }

        // Append one row
        fprintf(fp,
            "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
            nprocs,
            threads_per_rank,
            max_read,
            max_repl,
            max_compare,
            max_gather_offset,
            max_sum_offset,
            max_gather_rank,
            max_sum_rank,
            max_broadcast_rank,
            max_indicator,
            max_gather_sort,
            max_sum_sort
        );

        fclose(fp);
    }

    if(col_rank == 0) {
        for(size_t i=0; i < compRegionWidth; i++) {
            printCiphertext("S_" + std::to_string(i) , S[i], subVectorLength, rank, keyPair);
        }
    }
    return ctInputs;
}




std::vector<Ciphertext<DCRTPoly>> optimizedDistributedSortingWithCorrectionGridWise(    
    std::string caseName
)
{

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string dir = "/eagle/dist_relational_alg/chulu/fhe_test_cases/" + caseName;
    std::string jsonFile = dir + "/input.json";
    

    JsonConfig cfg = readJsonConfig(dir + "/input.json");
    
    // const usint compareDepth = 10;
    // const usint indicatorDepth = 10;
    const usint dg_c       = cfg.dg_c;
    const usint df_c       = cfg.df_c;
    const usint dg_i       = cfg.dg_i;
    const usint df_i       = cfg.df_i;

    const int subVectorLength = cfg.subVectorLength;
    const int numCiphertext = cfg.numCiphertext;
    int procWidth = static_cast<int>(std::sqrt(nprocs));
    const int compRegionWidth = numCiphertext / procWidth;
    const int vectorLength = numCiphertext * subVectorLength;
    
    std::cout << "Rank=" << rank 
              << " subVectorLength=" << cfg.subVectorLength
              << " numCiphertext=" << cfg.numCiphertext
              << " compRegionWidth=" << compRegionWidth
              << " procWidth=" << procWidth 
              << std::endl;


    int dims[2] = {procWidth, procWidth};
    int periods[2] = {0, 0};

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    int row = coords[0];
    int col = coords[1];
    
    MPI_Barrier(cart_comm);

    double t_read_start = MPI_Wtime();

    std::vector<Ciphertext<DCRTPoly>> ctInputs;

    for (int i = 0; i < compRegionWidth; i++) {
        size_t index = (col * compRegionWidth) + i;
        std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string(index) + ".txt";
        ctInputs.push_back(readChunk(ctFile));
        std::cout << "Ciphertext chunk " << index << " loaded successfully." << std::endl;
    }

    if (row != col) {
        for (int i = 0; i < compRegionWidth; i++) {
            size_t index = (row * compRegionWidth) + i;
            std::string ctFile = dir + "/ciphertext_chunk_" + std::to_string(index) + ".txt";
            ctInputs.push_back(readChunk(ctFile));
            std::cout << "Ciphertext chunk " << index << " loaded successfully." << std::endl;
        }
    }

    CryptoContext<DCRTPoly> cryptoContext = ctInputs[0]->GetCryptoContext();

    // std::cout << "Rank " << rank << " - Crypto address 0 : " << ctInputs[0]->GetCryptoContext().get() << std::endl;

    // std::cout << "Rank " << rank << " - Crypto address 1 : " << ctInputs[1]->GetCryptoContext().get() << std::endl;

    std::string evalMultFile = dir + "/key-eval-mult.txt";
    readEvalMultKey(cryptoContext, evalMultFile);
    std::cout << "Rank = " << rank << " - key eval multiplication loaded successfully" << std::endl;
    // print_memory_usage("after loading key multiplication");

    std::string evalRotFile = dir + "/key-eval-rot.txt";
    readEvalRotKey(cryptoContext, evalRotFile);
    // print_memory_usage("after loading key rotation");
    std::cout << "Rank = " << rank << " - key eval rotation loaded successfully" << std::endl;

    std::string secKeyFile = dir + "/key-secret.txt";
    KeyPair<DCRTPoly> keyPair = readKey(secKeyFile);
    std::cout << "Rank = " << rank << " - secret key loaded successfully" << std::endl;
    double t_read_end = MPI_Wtime();

    printDetailedMemoryUsage(rank, "after read");
    MPI_Barrier(cart_comm);

    // Phase 2: Replication
    double t_comp_repl = 0.0;
    double t_comm_repl = 0.0;

    std::vector<Ciphertext<DCRTPoly>> replR(compRegionWidth);
    std::vector<Ciphertext<DCRTPoly>> replC(compRegionWidth);

    int numLocalCiphertexts = ctInputs.size();

    MPI_Comm row_comm;
    MPI_Comm_split(cart_comm, row, col, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);


    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, col, row, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // Bug for case_1 nprocs=4
    if(compRegionWidth >= procWidth) {
        std::cout << "A" << std::endl;
        double t_comp_start = MPI_Wtime();
        #pragma omp parallel for collapse(2)
        for(int loopID = 0; loopID < 2; loopID++) {
            for(int j=0; j < compRegionWidth / procWidth; j++) {
                if(loopID == 0) {
                    replR[j] = replicateRow(ctInputs[row * (compRegionWidth / procWidth) + j], subVectorLength);
                } else {
                    replC[j] = replicateColumn(transposeRow(ctInputs[compRegionWidth % numLocalCiphertexts + col * (compRegionWidth / procWidth) + j], subVectorLength, true), subVectorLength);
                }
            }
        }
        std::cout << "Rank " << rank << "- Compute replication" << std::endl;
        double t_comp_end = MPI_Wtime();
        t_comp_repl += (t_comp_end - t_comp_start);
        
        double t_comm_start = MPI_Wtime();
        Communication::allGather(replC, compRegionWidth / procWidth, row_comm);
        std::cout << "Rank " << rank << "- allGather 1" << std::endl;
        Communication::allGather(replR, compRegionWidth / procWidth, col_comm);
        std::cout << "Rank " << rank << "- allGather 2" << std::endl;
        double t_comm_end = MPI_Wtime();
        t_comm_repl += (t_comm_end - t_comm_start);
        
    }
    else {
        std::cout << "B" << std::endl;

        int colCt = compRegionWidth % numLocalCiphertexts;
        std::vector<Ciphertext<DCRTPoly>> tempReplR(compRegionWidth);

        // -------------------------------------------------
        // COMM: create partial communicators
        // -------------------------------------------------
        double t_comm_start = MPI_Wtime();

        MPI_Comm partial_row_comm;
        int color = (col < compRegionWidth) ? row : MPI_UNDEFINED;
        MPI_Comm_split(cart_comm, color, col, &partial_row_comm);

        MPI_Comm partial_col_comm;
        color = (row >= procWidth - compRegionWidth) ? col : MPI_UNDEFINED;
        MPI_Comm_split(cart_comm, color, row, &partial_col_comm);

        MPI_Comm partial_row_comm_2;
        color = (col >= procWidth - compRegionWidth) ? row : MPI_UNDEFINED;
        int key = procWidth - 1 - col;
        MPI_Comm_split(cart_comm, color, key, &partial_row_comm_2);

        MPI_Comm special_col_comm;
        color = (col < compRegionWidth) ? col : MPI_UNDEFINED;
        key = row;
        if(col == procWidth - 1 && row < compRegionWidth) {
            color = row;
            key = procWidth;
        }
        MPI_Comm_split(cart_comm, color, key, &special_col_comm);

        double t_comm_end = MPI_Wtime();
        t_comm_repl += (t_comm_end - t_comm_start);
        
        // =================================================
        // (1) replC
        // =================================================
        if(col < compRegionWidth) {
            std::vector<Ciphertext<DCRTPoly>> repData(1);

            // ---- COMPUTE ----
            double t_comp_start = MPI_Wtime();
            repData[0] = replicateColumn(
                transposeRow(ctInputs[colCt + col], subVectorLength, true),
                subVectorLength
            );
            double t_comp_end = MPI_Wtime();
            t_comp_repl += (t_comp_end - t_comp_start);

            // ---- COMM ----
            std::vector<std::vector<Ciphertext<DCRTPoly>>> buffer;

            t_comm_start = MPI_Wtime();
            Communication::gatherVectorToRoot(repData, buffer, 0, partial_row_comm);
            t_comm_end = MPI_Wtime();
            t_comm_repl += (t_comm_end - t_comm_start);

            // ---- COMPUTE (copy on root) ----
            t_comp_start = MPI_Wtime();
            if(col == 0) {
                for(int i = 0; i < compRegionWidth; i++) {
                    replC[i] = buffer[i][0];
                }
            }
            t_comp_end = MPI_Wtime();
            t_comp_repl += (t_comp_end - t_comp_start);
        }
        std::cout << "Rank " << rank << "- Gather vector 1" << std::endl;
        // =================================================
        // (2) replR
        // =================================================
        if(row >= procWidth - compRegionWidth && col >= compRegionWidth) {
            std::vector<Ciphertext<DCRTPoly>> repData(1);

            // ---- COMPUTE ----
            double t_comp_start = MPI_Wtime();
            repData[0] = replicateRow(
                ctInputs[compRegionWidth - (procWidth - row)],
                subVectorLength
            );
            double t_comp_end = MPI_Wtime();
            t_comp_repl += (t_comp_end - t_comp_start);

            // ---- COMM ----
            std::vector<std::vector<Ciphertext<DCRTPoly>>> buffer;

            t_comm_start = MPI_Wtime();
            Communication::gatherVectorToRoot(repData, buffer,
                                            compRegionWidth - 1,
                                            partial_col_comm);
            t_comm_end = MPI_Wtime();
            t_comm_repl += (t_comm_end - t_comm_start);

            // ---- COMPUTE (copy on root) ----
            t_comp_start = MPI_Wtime();
            if(row == procWidth - 1) {
                for(int i = 0; i < compRegionWidth; i++) {
                    replR[i] = buffer[i][0];
                }
            }
            t_comp_end = MPI_Wtime();
            t_comp_repl += (t_comp_end - t_comp_start);
        }
        std::cout << "Rank " << rank << "- Gather vector 2" << std::endl;
        // =================================================
        // (3) tempReplR (reverse copy)
        // =================================================
        if(row < compRegionWidth && col >= procWidth - compRegionWidth) {
            std::vector<Ciphertext<DCRTPoly>> repData(1);

            // ---- COMPUTE ----
            double t_comp_start = MPI_Wtime();
            repData[0] = replicateRow(
                ctInputs[colCt + (col % compRegionWidth)],
                subVectorLength
            );
            double t_comp_end = MPI_Wtime();
            t_comp_repl += (t_comp_end - t_comp_start);

            // ---- COMM ----
            std::vector<std::vector<Ciphertext<DCRTPoly>>> buffer;

            t_comm_start = MPI_Wtime();
            Communication::gatherVectorToRoot(repData, buffer,
                                            0,
                                            partial_row_comm_2);
            t_comm_end = MPI_Wtime();
            t_comm_repl += (t_comm_end - t_comm_start);

            // ---- COMPUTE (reverse copy) ----
            t_comp_start = MPI_Wtime();
            if(col == procWidth - 1) {
                for(int i = 0; i < compRegionWidth; i++) {
                    tempReplR[i] = buffer[compRegionWidth - i - 1][0];
                }
            }
            t_comp_end = MPI_Wtime();
            t_comp_repl += (t_comp_end - t_comp_start);
        }

        std::cout << "Rank " << rank << "- Gather vector 3" << std::endl;

        // =================================================
        // (4) Broadcast
        // =================================================
        t_comm_start = MPI_Wtime();

        Communication::broadcastVectorFromRoot(replC, 0, row_comm);
        std::cout << "Rank " << rank << "- Broadcast vector 1" << std::endl;
        Communication::broadcastVectorFromRoot(replR, procWidth - 1, col_comm);
        std::cout << "Rank " << rank << "- Broadcast vector 2" << std::endl;

        if(special_col_comm != MPI_COMM_NULL) {
            int s; MPI_Comm_size(special_col_comm, &s);
            if (procWidth >= s) {
                std::cerr << "BAD ROOT: procWidth=" << procWidth
                        << " comm_size=" << s
                        << " global rank=" << rank << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            Communication::broadcastVectorFromRoot(tempReplR,
                                                procWidth,
                                                special_col_comm);
        }
        std::cout << "Rank " << rank << "- Broadcast vector 3" << std::endl;

        t_comm_end = MPI_Wtime();
        t_comm_repl += (t_comm_end - t_comm_start);

        // =================================================
        // (5) Final overwrite
        // =================================================
        double t_comp_start = MPI_Wtime();
        if(col < compRegionWidth) {
            for(int i = 0; i < compRegionWidth; i++) {
                replR[i] = tempReplR[i];
            }
        }
        std::cout << "Rank " << rank << "- Final overwrite" << std::endl;
        double t_comp_end = MPI_Wtime();
        t_comp_repl += (t_comp_end - t_comp_start);
    }

    



    if(rank == 0) {
            std::cout << "Rank " << rank << " replR levels: " << replR[0]->GetLevel() << std::endl;
            std::cout << "Rank " << rank << " replC levels: " << replC[0]->GetLevel() << std::endl;
    }

    std::cout << "Rank " << rank << "- finish replication" << std::endl;

    
    printDetailedMemoryUsage(rank, "after replication");

    MPI_Barrier(cart_comm);

    // PHASE 3: COMPARE
    double t_compare_start = MPI_Wtime();
    

    std::vector<Ciphertext<DCRTPoly>> R(compRegionWidth);
    std::vector<bool> Rinitialized(compRegionWidth);
    

    std::vector<Ciphertext<DCRTPoly>> Ev(compRegionWidth);
    std::vector<bool> Evinitialized(compRegionWidth);


    std::vector<double> triangleMask(subVectorLength * subVectorLength, 1.0);
    for(int i=0; i<subVectorLength; i++) {
        for(int j=0; j<subVectorLength; j++) {
            if(i < j) {
                triangleMask[i * subVectorLength + j] = 0.0;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i=0; i < compRegionWidth; i++) {
        for(int j=0; j < compRegionWidth; j++) {
        Ciphertext<DCRTPoly> Cij = compareAdv(
                                    replC[i], replR[j],
                                    dg_c, df_c
                                );
            // Ciphertext<DCRTPoly> Cij = compare(
            //     replC[i], replR[j], -1, 1, depth2degree(compareDepth)
            // );

            // std::cout << "[C_" << i << j << "] Rank " << rank << " ct = " << Cij << std::endl;

            Ciphertext<DCRTPoly> Eij = 4 * (1 - Cij) * Cij;
        

            Ciphertext<DCRTPoly> Pij = Eij;
            
            if(row < col) {
                Pij = Eij * 0;
            }
            else if(row == col) {
                if(i < j) {
                    Pij = Eij * 0;
                }
                else if(i == j) {
                    Pij = Eij * triangleMask;
                }
            }
            
            Eij = Eij * 0.5;
            
            Eij = Pij - Eij;
            // std::cout << "[E_" << i << j <<"] Rank " << rank << " ct = " << Eij << std::endl;

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
            
            #pragma omp critical
            {
                if(!Evinitialized[i]) {
                    Ev[i] = Eij;
                    Evinitialized[i] = true;
                }
                else {
                    Ev[i] = Ev[i] + Eij;
                }
            }
        }    
    }

    if(rank == 0) {
            std::cout << "Rank " << rank << " R levels: " << R[0]->GetLevel() << std::endl;
            std::cout << "Rank " << rank << " Ev levels: " << Ev[0]->GetLevel() << std::endl;
    }

    double t_compare_end = MPI_Wtime();
    std::cout << "Rank " << rank << "- finish comparison" << std::endl;


    MPI_Barrier(cart_comm);

    // PHASE 4: Gather offset results to root
    double t_gather_offset_start = MPI_Wtime();

    std::vector<std::vector<Ciphertext<DCRTPoly>>> allEv;
     
    Communication::gatherVectorToRoot(Ev, allEv, 0, row_comm);

    double t_gather_offset_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather offset" << std::endl;

    MPI_Barrier(cart_comm);


    // PHASE 5: Sum offset results to root
    double t_sum_offset_start = MPI_Wtime();

    if(row_rank == 0) {
        // #pragma omp parallel for collapse(2)
        #pragma omp parallel for
        for(int i=0; i < compRegionWidth; i++) {
            for(int j=1; j < row_size; j++) {
                // #pragma omp critical
                Ev[i] = Ev[i] + allEv[j][i];
                
            }        
        }
        
        #pragma omp parallel for
        for(int i=0; i < compRegionWidth; i++) {
                
            Ev[i] = sumColumns(Ev[i], subVectorLength, true);

            // std::cout << "[Ev_" << i << "] Rank " << rank << " ct = " << Ev[i] << std::endl;
            Ev[i] = replicateColumn(transposeRow(Ev[i], subVectorLength, true), subVectorLength);
            Ev[i] = Ev[i] - 0.5;
            
            // std::cout << "[R_" << i << "] Rank " << rank << " ct = " << R[i] << std::endl;
        }
    }

    double t_sum_offset_end = MPI_Wtime();

    if(rank == 0) {
            std::cout << "Rank " << rank << " R_after levels: " << R[0]->GetLevel() << std::endl;
            std::cout << "Rank " << rank << " Ev_after levels: " << Ev[0]->GetLevel() << std::endl;
    }

    std::cout << "Rank " << rank << "- finish sum offset" << std::endl;



    MPI_Barrier(cart_comm);

    // PHASE 6: Gather rank to root
    double t_gather_rank_start = MPI_Wtime();

    std::vector<std::vector<Ciphertext<DCRTPoly>>> allRanks;

    Communication::gatherVectorToRoot(R, allRanks, 0, row_comm);

    double t_gather_rank_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather rank" << std::endl;

    MPI_Barrier(cart_comm);

    // PHASE 7: Sum to compute rank on each row

    double t_sum_rank_start = MPI_Wtime();
    if(row_rank == 0) {
        
        // #pragma omp parallel for collapse(2)
        #pragma omp parallel for
        for(int i=0; i<compRegionWidth; i++) {
            for(int j=1; j< row_size; j++) {
                // #pragma omp critical
                R[i] = R[i] + allRanks[j][i];
            }
        }
        

        // R[0] = R[0] + 0.5;

        #pragma omp parallel for
        for(int i=0; i < compRegionWidth; i++) {
            // R[i] = R[i] + 0.5;
            R[i] = sumColumns(R[i], subVectorLength, true);
            R[i] = replicateColumn(R[i], subVectorLength);
            R[i] = R[i] + 0.5;

            R[i] = R[i] + Ev[i];
            // std::cout << "[R_" << i << "] Rank " << rank << " ct = " << R[i] << std::endl;
        }
        
        // for(size_t i=0; i < compRegionWidth; i++) {
        //     printCiphertext("R", R[i], subVectorLength, rank, keyPair);
        // }
        
        if(rank == 0) {
                std::cout << "Rank " << rank << " R_final levels: " << R[0]->GetLevel() << std::endl;
        }
    }

    double t_sum_rank_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish sum rank" << std::endl;

    MPI_Barrier(cart_comm);

    // PHASE 8: Broadcast ranking results
    double t_broadcast_rank_start = MPI_Wtime();
    Communication::broadcastVectorFromRoot<Ciphertext<DCRTPoly>>(R, 0, row_comm);
    double t_broadcast_rank_end = MPI_Wtime();

    MPI_Barrier(cart_comm);

    // PHASE 9: Indicator

    double t_indicator_start = MPI_Wtime();
    std::vector<std::vector<double>> subMasks(compRegionWidth, std::vector<double>(subVectorLength * subVectorLength));

    for(int i = 0; i < compRegionWidth; i++) {
        for(int j = 0; j < subVectorLength; j++) {
            for(int k = 0; k < subVectorLength; k++) {
                // We need to adjust this if we set rank 0 as client!
                subMasks[i][j * subVectorLength + k] = -static_cast<double>((col * subVectorLength * compRegionWidth) + (i * subVectorLength) + k + 1);
            }
        }
    }


    std::vector<Ciphertext<DCRTPoly>> S(compRegionWidth);
    std::vector<bool> Sinitialized(compRegionWidth);

    #pragma omp parallel for collapse(2)
    for(int j=0; j < compRegionWidth; j++) {
        for(int i=0; i < compRegionWidth; i++) {
            Ciphertext<DCRTPoly> Cij = indicatorAdv(
                R[i] + subMasks[j],
                vectorLength,
                dg_i, df_i
            ) * replC[i];

            #pragma omp critical 
            {
                if(!Sinitialized[j]) {
                    S[j] = Cij;
                    Sinitialized[j] = true;
                }
                else {
                    S[j] = S[j] + Cij;
                }
            }
        }
        // std::cout << "[ReplC_" << i  << "] Rank " << rank << " ct = " << replC[i] << std::endl;    
    }

    // for(size_t i=0; i < compRegionWidth; i++) {
    //     printCiphertext("S_before_" + std::to_string(i), S[i], subVectorLength, rank, keyPair);
    // }

    if(rank == 0) {
            std::cout << "Rank " << rank << " S levels: " << S[0]->GetLevel() << std::endl;
    }

    double t_indicator_end = MPI_Wtime();
    std::cout << "Rank " << rank << "- finish indicator" << std::endl;

    MPI_Barrier(cart_comm);


    // PHASE 10: Gather sort
    double t_gather_sort_start = MPI_Wtime();
    Communication::gatherVectorToRoot(S, allRanks, 0, col_comm);
    double t_gather_sort_end = MPI_Wtime();

    std::cout << "Rank " << rank << "- finish gather sort" << std::endl;

    MPI_Barrier(cart_comm);


    // PHASE 11: Sum sort
    double t_sum_sort_start = MPI_Wtime();
    if(col_rank == 0) {
        // std::cout << "all rank size = " << allRanks.size() << std::endl;

        #pragma omp parallel for
        for(int i=0; i < compRegionWidth; i++) {
            for(int j=1; j < row_size; j++) {
                #pragma omp critical
                S[i] = S[i] + allRanks[j][i];
            }
        }
        
        #pragma omp parallel for
        for(int i=0; i < compRegionWidth; i++) {
            S[i] = sumRows(S[i], subVectorLength);
        }

        // for(size_t i=0; i < compRegionWidth; i++) {
        //     printCiphertext("S_" + std::to_string(i) , S[i], subVectorLength, rank, keyPair);
        // }
    }

    double t_sum_sort_end = MPI_Wtime();

    if(rank == 0) {
            std::cout << "Rank " << rank << " S_final levels: " << S[0]->GetLevel() << std::endl;
    }

    std::cout << "Rank " << rank << "- finish sum sort indicator" << std::endl;

    double t_read = t_read_end - t_read_start;
    // double t_repl = t_repl_end - t_repl_start;
    double t_compare = t_compare_end - t_compare_start;
    double t_gather_rank = t_gather_rank_end - t_gather_rank_start;

    double t_gather_offset = t_gather_offset_end - t_gather_offset_start;
    double t_sum_offset = t_sum_offset_end - t_sum_offset_start;

    double t_sum_rank = t_sum_rank_end - t_sum_rank_start;
    double t_broadcast_rank = t_broadcast_rank_end - t_broadcast_rank_start;
    double t_indicator = t_indicator_end - t_indicator_start;
    double t_gather_sort = t_gather_sort_end - t_gather_sort_start;
    double t_sum_sort = t_sum_sort_end - t_sum_sort_start;

    double max_read, t_comp_repl_max, t_comm_repl_max, max_compare, max_gather_rank, max_sum_rank, max_gather_offset, max_sum_offset;
    double max_broadcast_rank, max_indicator, max_gather_sort, max_sum_sort;

    // ---------------------------------------------------------
    // Perform MAX reduction for each timing phase
    // ---------------------------------------------------------
    MPI_Reduce(&t_read,           &max_read,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_repl, &t_comp_repl_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&t_comm_repl, &t_comm_repl_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&t_compare,        &max_compare,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    

    MPI_Reduce(&t_gather_offset,    &max_gather_offset,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_offset,       &max_sum_offset,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t_gather_rank,    &max_gather_rank,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_rank,       &max_sum_rank,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_broadcast_rank, &max_broadcast_rank, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_indicator,      &max_indicator,      1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather_sort,    &max_gather_sort,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_sum_sort,       &max_sum_sort,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        
        printf("\n=== Aggregated Max Timing (seconds) ===\n");
        printf("read           : %f\n", max_read);
        printf("repl           : %f\n", t_comp_repl_max);
        printf("comm_repl      : %f\n", t_comm_repl_max);
        printf("compare        : %f\n", max_compare);
        printf("gather_offset  : %f\n", max_gather_offset);
        printf("sum_offset     : %f\n", max_sum_offset);
        printf("gather_rank    : %f\n", max_gather_rank);
        printf("sum_rank       : %f\n", max_sum_rank);
        printf("broadcast_rank : %f\n", max_broadcast_rank);
        printf("indicator      : %f\n", max_indicator);
        printf("gather_sort    : %f\n", max_gather_sort);
        printf("sum_sort       : %f\n", max_sum_sort);
        printf("=======================================\n");

        int threads_per_rank = 0;
        #pragma omp parallel
        {
            #pragma omp single
            threads_per_rank = omp_get_num_threads();
        }

        const char *filename = "timing.csv";

        FILE *fp = fopen(filename, "a+");   // create if not exist, append if exist
        if (!fp) {
            perror("fopen");

        }

        // Check if file is empty
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);

        if (size == 0) {
            // Write header only once
            fprintf(fp,
                "nranks,threads_per_rank,read,repl,repl_comm,compare,gather_offset,sum_offset,"
                "gather_rank,sum_rank,broadcast_rank,"
                "indicator,gather_sort,sum_sort\n");
        }

        // Append one row
        fprintf(fp,
            "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
            nprocs,
            threads_per_rank,
            max_read,
            t_comp_repl_max,
            t_comm_repl_max,
            max_compare,
            max_gather_offset,
            max_sum_offset,
            max_gather_rank,
            max_sum_rank,
            max_broadcast_rank,
            max_indicator,
            max_gather_sort,
            max_sum_sort
        );

        fclose(fp);
    }
    
    printDetailedMemoryUsage(rank, "Done!");

    if(col_rank == 0) {
        for(int i=0; i < compRegionWidth; i++) {
            printCiphertext("S_" + std::to_string(i) , S[i], subVectorLength, rank, keyPair);
        }
    }
    return ctInputs;
}
