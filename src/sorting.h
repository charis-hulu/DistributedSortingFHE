#include <mpi.h>
#include <vector>
#include "utils-io.h"
#include "utils-eval.h"
#include "utils-matrices.h"
// #include "utils-communication.h"

std::vector<Ciphertext<DCRTPoly>> distributedSortingGridWise(    
    std::string caseName
);

std::vector<Ciphertext<DCRTPoly>> distributedSortingWithCorrectionGridWise(    
    std::string caseName
);

std::vector<Ciphertext<DCRTPoly>> optimizedDistributedSortingWithCorrectionGridWise(    
    std::string caseName
);

std::vector<Ciphertext<DCRTPoly>> sortingWithCorrectionGridWise(    
    std::string caseName
);