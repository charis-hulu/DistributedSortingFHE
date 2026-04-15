#include <iostream>
#include "sorting.h"

#include <openfhe.h>
#include <omp.h>


int main(int argc, char**argv) {


    MPI_Init(&argc, &argv);

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

    int maxThreads = omp_get_max_threads();
    std::cout << "MPI rank " << rank << ": OMP_MAX_THREADS = " << maxThreads << std::endl;
    omp_set_num_threads(maxThreads);
    std::cout << "nested omp on" << std::endl;
    omp_set_max_active_levels(10);

    std::string caseName = argv[1];

    std::vector<Ciphertext<DCRTPoly>> c = optimizedDistributedSortingWithCorrectionGridWise(caseName);

    // Example: print loaded ciphertexts
    // for (size_t i = 0; i < c.size(); i++) {
    //     std::cout << "c[" << i << "] = " << c[i] << std::endl;
    // }

    MPI_Finalize();

}