// mpi_utils.hpp
#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <sstream>
#include "cereal/archives/binary.hpp"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
// #include "openfhe.h"
// #include "utils/serial.h"

// CEREAL_REGISTER_TYPE(lbcrypto::CryptoParametersCKKSRNS);
// CEREAL_REGISTER_TYPE(lbcrypto::SWITCHCKKSRNS);
// CEREAL_REGISTER_TYPE(lbcrypto::SchemeCKKSRNS);
// CEREAL_REGISTER_TYPE(lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>)
// CEREAL_REGISTER_TYPE(lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>);

// CEREAL_REGISTER_TYPE(lbcrypto::KeyPair<lbcrypto::DCRTPoly>)

namespace Communication {
    
    template <typename T>
    std::string serializeData(T &data) {
        // cout << "SERIALIZING DATA..." << endl;
        std::stringstream ss;
        { //Needed for RAII in Cereal
            cereal::BinaryOutputArchive archive( ss );
            archive( data );
            // archive(data);
        }
        std::string serialized = ss.str();
        return serialized;
    }

    template <typename T>
    T deserializeData(const std::string& serialized) {
        T data;
        std::stringstream ss(serialized);
        {
            cereal::BinaryInputArchive archive(ss);
            archive(data);
        }
        return data;
    }


    void broadcastString(std::string &msg, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        int length = msg.size();

        // Step 1: broadcast length of the string
        MPI_Bcast(&length, 1, MPI_INT, root, comm);

        // Step 2: prepare buffer for string data
        std::cout << "Buffer size = " << length << std::endl;
        std::vector<char> buffer(length);
        std::cout << "Maximum buffer = " << buffer.max_size() << std::endl;



        if(rank == root) {
            std::copy(msg.begin(), msg.end(), buffer.begin());
        }

        // Step 3: broadcast the string content 
        MPI_Bcast(buffer.data(), length, MPI_CHAR, root, comm);
        // std::cout << "Rank " << rank << "Finish broad cast data" << std::endl;
    

        // Step 4: reconstruct string on non-root processes
        if(rank != root) {
            msg = std::string(buffer.begin(), buffer.end());
        }
    }

    template <typename T>
    void broadcastData(T &data, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        std::string serialized;

        int rank;
        MPI_Comm_rank(comm, &rank);

        if(rank == root) {
            // Serialize on root
            serialized = serializeData(data);
        }

        // Broadcast the serialized string
        broadcastString(serialized, root, comm);

        if(rank != root) {
            // Deserialize on non-root
            data = deserializeData<T>(serialized);
        }
    }

    

    template <typename T>
    void broadcastNumericVector(std::vector<T> &vec, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        // Step 1: broadcast the size of the vector
        int size = vec.size();
        MPI_Bcast(&size, 1, MPI_INT, root, comm);

        // Step 2: resize vector on non-root ranks
        if(rank != root) {
            vec.resize(size);
        }

        // Step 3: broadcast the raw data
        MPI_Bcast(vec.data(), size, std::is_same<T,int>::value ? MPI_INT : MPI_DOUBLE, root, comm);
    }



    void broadcastStringVector(std::vector<std::string>& vec, int root, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        // Step 1: broadcast size of vector
        int count = vec.size();
        MPI_Bcast(&count, 1, MPI_INT, root, comm);

        if (rank != root) {
            vec.resize(count);
        }

        // Step 2: broadcast each string size & string data
        for (int i = 0; i < count; i++) {
            int len = vec[i].size();
            MPI_Bcast(&len, 1, MPI_INT, root, comm);

            if (rank != root) {
                vec[i].resize(len);
            }

            MPI_Bcast(vec[i].data(), len, MPI_CHAR, root, comm);
        }
    }
    
    void sendString(const std::string &msg, int dest, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        int length = msg.size();
        // Step 1: send length first
        MPI_Send(&length, 1, MPI_INT, dest, tag, comm);
        // Step 2: send actual string data
        MPI_Send(msg.c_str(), length, MPI_CHAR, dest, tag + 1, comm);
    }

    // Receive a string from src
    void recvString(std::string &buffer, int src, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        int length;
        MPI_Recv(&length, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE); // receive length

        buffer.resize(length); // make sure buffer is large enough
        MPI_Recv(buffer.data(), length, MPI_CHAR, src, tag + 1, comm, MPI_STATUS_IGNORE); // receive data
    }

    // template <class T>
    // void distributeRotationKeys(T cryptoContext, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    //     int rank;
    //     MPI_Comm_rank(comm, &rank);

    //     std::string serKeys;

    //     if (rank == root) {
    //         std::stringstream ss;
    //         std::cout << "Serialize automorphism key..." << std::endl; 
    //         cryptoContext->SerializeEvalAutomorphismKey(ss, SerType::BINARY);
    //         serKeys = ss.str();
    //     }

    //     // Broadcast using your function
    //     broadcastString(serKeys, root, comm);
    //     std::cout << "Broadcast serialized automorphism key..." << std::endl;

    //     // All ranks load keys
    //     std::stringstream ssIn(serKeys);
    //     cryptoContext->DeserializeEvalAutomorphismKey(ssIn, SerType::BINARY);
    //     std::cout << "Deserialize automorphism key..." << std::endl;
    // }
    template <class T>
    void distributeRotationKeys(T cryptoContext, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        std::string serializedData;

        if (rank == root) {
            std::ostringstream oss;
            cryptoContext->SerializeEvalAutomorphismKey(oss, SerType::BINARY);
            serializedData = oss.str();
        }

        size_t dataSize = serializedData.size();
        MPI_Bcast(&dataSize, 1, MPI_UNSIGNED_LONG_LONG, root, comm);

        if (rank != root)
            serializedData.resize(dataSize);

        MPI_Bcast(serializedData.data(), dataSize, MPI_BYTE, root, comm);

        if (rank != root) {
            std::istringstream iss(serializedData);
            cryptoContext->DeserializeEvalAutomorphismKey(iss, SerType::BINARY);
        }

        MPI_Barrier(comm);
    }

    template <typename T>
    void distributeMultKeys(T cryptoContext, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        std::string serializedData;

        if (rank == root) {
            // Serialize EvalMultKeys in memory
            std::ostringstream oss;
            cryptoContext->SerializeEvalMultKey(oss, SerType::BINARY);
            serializedData = oss.str();
        }

        // Step 1: Broadcast the serialized data size
        size_t dataSize = serializedData.size();
        MPI_Bcast(&dataSize, 1, MPI_UNSIGNED_LONG_LONG, root, comm);

        // Step 2: Broadcast the serialized data itself
        if (rank != root)
            serializedData.resize(dataSize);

        MPI_Bcast(serializedData.data(), dataSize, MPI_BYTE, root, comm);

        // Step 3: Deserialize on non-root ranks
        if (rank != root) {
            std::istringstream iss(serializedData);
            cryptoContext->DeserializeEvalMultKey(iss, SerType::BINARY);
        }

        MPI_Barrier(comm);
    }

    template <typename T>
    void allGatherV(T &data, std::vector<T> &all_buffer, MPI_Comm comm = MPI_COMM_WORLD) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Serialize your data to string
        std::string serialized = serializeData(data);

        // Get size of this string
        int send_size = serialized.size();

        // 1. Gather all sizes
        std::vector<int> recv_sizes(size);
        MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm);

        // 2. Compute displacements
        std::vector<int> displs(size, 0);
        int total_size = recv_sizes[0];
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recv_sizes[i-1];
            total_size += recv_sizes[i];
        }

        // 3. Prepare receive buffer
        std::vector<char> recvbuf(total_size);

        // 4. Allgatherv: gather variable-length strings
        MPI_Allgatherv(serialized.data(), send_size, MPI_CHAR,
                    recvbuf.data(), recv_sizes.data(), displs.data(), MPI_CHAR, comm);

        // 5. Deserialize all strings
        all_buffer.clear();
        for (int i = 0; i < size; i++) {
            std::string s(&recvbuf[displs[i]], recv_sizes[i]);
            all_buffer.push_back(deserializeData<T>(s));
        }
    }

    // template <typename T>
    // void allGather(std::vector<T> &data,
    //             int send_count,
    //             MPI_Comm comm)
    // {
    //     int rank, size;
    //     MPI_Comm_rank(comm, &rank);
    //     MPI_Comm_size(comm, &size);

    
    //     std::vector<T> local(data.begin(), data.begin() + send_count);
    //     std::string serialized = serializeData(local);
    //     int send_size = serialized.size();

  
    //     std::vector<int> recv_sizes(size);
    //     MPI_Allgather(&send_size, 1, MPI_INT,
    //                 recv_sizes.data(), 1, MPI_INT,
    //                 comm);

     
    //     std::vector<int> displs(size);
    //     int total_size = 0;
    //     for (int i = 0; i < size; i++) {
    //         displs[i] = total_size;
    //         total_size += recv_sizes[i];
    //     }

    //     std::vector<char> recvbuf(total_size);

    //     MPI_Allgatherv(
    //         serialized.data(), send_size, MPI_CHAR,
    //         recvbuf.data(), recv_sizes.data(), displs.data(), MPI_CHAR,
    //         comm
    //     );


    //     for (int i = 0; i < size; i++) {
    //         std::string s(&recvbuf[displs[i]], recv_sizes[i]);
    //         auto part = deserializeData<std::vector<T>>(s);

    //         std::copy(part.begin(), part.end(),
    //                 data.begin() + i * send_count);
    //     }
    // }

    template <typename T>
    void allGather(std::vector<T> &data,
                int send_count,
                MPI_Comm comm)
    {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (send_count < 0) throw std::invalid_argument("send_count < 0");

        // Ensure data can hold all gathered pieces
        const size_t needed = static_cast<size_t>(size) * static_cast<size_t>(send_count);
        if (data.size() < needed) data.resize(needed);

        // 1) Serialize exactly send_count items (as you intended)
        if (static_cast<size_t>(send_count) > data.size()) {
            throw std::out_of_range("send_count exceeds local data size");
        }

        std::vector<T> local(data.begin(), data.begin() + send_count);
        std::string serialized = serializeData(local);

        long long local_bytes_ll = static_cast<long long>(serialized.size());

        // 2) Max-reduce for padding size
        long long max_bytes_ll = 0;
        MPI_Allreduce(&local_bytes_ll, &max_bytes_ll, 1, MPI_LONG_LONG, MPI_MAX, comm);

        if (max_bytes_ll > static_cast<long long>(std::numeric_limits<int>::max())) {
            // Same limitation: MPI_Allgather counts are int in standard collectives
            if (rank == 0) {
                // don't assume root; still useful message
                fprintf(stderr,
                        "Error: max serialized size (%lld) > INT_MAX; need chunking or MPI-4 large-count collectives.\n",
                        max_bytes_ll);
            }
            MPI_Abort(comm, 2);
        }

        const int max_bytes = static_cast<int>(max_bytes_ll);

        // 3) Collect true sizes (int is safe because each <= max_bytes <= INT_MAX)
        int local_bytes_int = static_cast<int>(serialized.size());
        std::vector<int> recv_sizes(size);
        MPI_Allgather(&local_bytes_int, 1, MPI_INT,
                    recv_sizes.data(), 1, MPI_INT,
                    comm);

        // 4) Pad and Allgather fixed-size blocks
        std::vector<char> sendbuf(static_cast<size_t>(max_bytes), '\0');
        if (!serialized.empty()) {
            std::memcpy(sendbuf.data(), serialized.data(), static_cast<size_t>(local_bytes_int));
        }

        std::vector<char> recvbuf(static_cast<size_t>(size) * static_cast<size_t>(max_bytes));
        MPI_Allgather(sendbuf.data(), max_bytes, MPI_CHAR,
                    recvbuf.data(), max_bytes, MPI_CHAR,
                    comm);

        // 5) Deserialize each rank’s true payload and write into output layout
        for (int i = 0; i < size; i++) {
            const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(max_bytes);
            const int true_n = recv_sizes[i];

            std::string s(recvbuf.data() + offset,
                        recvbuf.data() + offset + static_cast<size_t>(true_n));

            auto part = deserializeData<std::vector<T>>(s);

            // Optional sanity: ensure each rank produced exactly send_count items
            if (static_cast<int>(part.size()) != send_count) {
                // If you prefer hard-fail:
                // throw std::runtime_error("deserialize size mismatch");
                // For MPI context, abort is often clearer:
                fprintf(stderr,
                        "Rank %d: deserialized part from rank %d has %zu items, expected %d\n",
                        rank, i, part.size(), send_count);
                MPI_Abort(comm, 3);
            }

            std::copy(part.begin(), part.end(),
                    data.begin() + static_cast<size_t>(i) * static_cast<size_t>(send_count));
        }
    }
    // template <typename T>
    // void allGather(T &data, std::vector<T> &all_buffer, MPI_Comm comm = MPI_COMM_WORLD) {
    //     int rank, size;
    //     MPI_Comm_rank(comm, &rank);
    //     MPI_Comm_size(comm, &size);

    //     std::string serialized = serializeData(data);
    //     int send_size = serialized.size();


    //     std::vector<int> recv_sizes(size);
    //     MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, comm);


    //     std::vector<char> recvbuf(size * send_size);


    //     MPI_Allgather(serialized.data(), send_size, MPI_CHAR,
    //                 recvbuf.data(), send_size, MPI_CHAR, comm);


    //     all_buffer.clear();
    //     for (int i = 0; i < size; i++) {
    //         std::string s(&recvbuf[i * send_size], send_size);
    //         all_buffer.push_back(deserializeData<T>(s));
    //     }
    // }


    // template <typename T>
    // void gatherVectorToRoot(std::vector<T> &data,
    //                 std::vector<std::vector<T>> &all_buffer,
    //                 int root = 0,
    //                 MPI_Comm comm = MPI_COMM_WORLD)
    // {
    //     int rank, size;
    //     MPI_Comm_rank(comm, &rank);
    //     MPI_Comm_size(comm, &size);

    //     // Serialize local vector
    //     std::string serialized = serializeData(data);

    //     long long serialized_size = serialized.size();
        
    //     int send_size = serialized.size();

    //     // Allocate receive buffer only on root
    //     std::vector<char> recvbuf;
    //     if (rank == root) recvbuf.resize(size * send_size);
    //     std::cout << "Recvbuffer size = " <<  recvbuf.size() << std::endl;
    //     std::cout << "Rank " << rank << " - Serialized size = " << serialized_size << std::endl; 
    //     std::cout << "Rank " << rank << " - Send size = " << send_size << std::endl;

    //     // Gather serialized data to root
    //     MPI_Gather(
    //         serialized.data(), send_size, MPI_CHAR,
    //         recvbuf.data(), send_size, MPI_CHAR,
    //         root, comm
    //     );

    //     // Deserialize only on root
    //     if (rank == root) {
    //         all_buffer.clear();
    //         all_buffer.resize(size);
    //         for (int i = 0; i < size; i++) {
    //             std::string s(&recvbuf[i * send_size], send_size);
    //             all_buffer[i] = deserializeData<std::vector<T>>(s);
    //         }
    //     }
    // }

    template <typename T>
    void gatherVectorToRoot(std::vector<T> &data,
                            std::vector<std::vector<T>> &all_buffer,
                            int root = 0,
                            MPI_Comm comm = MPI_COMM_WORLD)
    {
        int rank, comm_size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);

        // Serialize local vector
        std::string serialized = serializeData(data);

        long long local_bytes_ll = static_cast<long long>(serialized.size());

        // 1) Max size across ranks
        long long max_bytes_ll = 0;
        MPI_Allreduce(&local_bytes_ll, &max_bytes_ll, 1, MPI_LONG_LONG, MPI_MAX, comm);

        if (max_bytes_ll > static_cast<long long>(std::numeric_limits<int>::max())) {
            if (rank == root) {
                std::cerr << "Error: max serialized size (" << max_bytes_ll
                        << ") > INT_MAX; need chunking or MPI-4 large-count collectives.\n";
            }
            MPI_Abort(comm, 2);
        }
        const int max_bytes = static_cast<int>(max_bytes_ll);

        // 2) Gather the true sizes on root (so root knows how many bytes to deserialize)
        int local_bytes_int = static_cast<int>(serialized.size());
        std::vector<int> recv_sizes;
        if (rank == root) recv_sizes.resize(comm_size);

        MPI_Gather(&local_bytes_int, 1, MPI_INT,
                rank == root ? recv_sizes.data() : nullptr, 1, MPI_INT,
                root, comm);

        // 3) Pad to max_bytes and gather fixed-size blocks to root
        std::vector<char> sendbuf(static_cast<size_t>(max_bytes), '\0');
        if (!serialized.empty()) {
            std::memcpy(sendbuf.data(), serialized.data(), static_cast<size_t>(local_bytes_int));
        }

        std::vector<char> recvbuf;
        if (rank == root) {
            recvbuf.resize(static_cast<size_t>(comm_size) * static_cast<size_t>(max_bytes));
        }

        MPI_Gather(sendbuf.data(), max_bytes, MPI_CHAR,
                rank == root ? recvbuf.data() : nullptr, max_bytes, MPI_CHAR,
                root, comm);

        // 4) Deserialize on root
        if (rank == root) {
            all_buffer.clear();
            all_buffer.resize(comm_size);

            for (int i = 0; i < comm_size; i++) {
                const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(max_bytes);
                const int true_n = recv_sizes[i];

                std::string s(recvbuf.data() + offset,
                            recvbuf.data() + offset + static_cast<size_t>(true_n));

                all_buffer[i] = deserializeData<std::vector<T>>(s);
            }
        }
    }
    // template <typename T>
    // void gatherVectorToRoot(std::vector<T> &data,
    //                         std::vector<std::vector<T>> &all_buffer,
    //                         int root = 0,
    //                         MPI_Comm comm = MPI_COMM_WORLD)
    // {
    //     int rank, comm_size;
    //     MPI_Comm_rank(comm, &rank);
    //     MPI_Comm_size(comm, &comm_size);

    //     // 1) Serialize local vector
    //     std::string serialized = serializeData(data);

    //     // local size as 64-bit for the max-reduction
    //     long long local_size_ll = static_cast<long long>(serialized.size());

    //     // 2) Allreduce max to get global maximum serialized size
    //     long long max_size_ll = 0;
    //     MPI_Allreduce(&local_size_ll, &max_size_ll, 1, MPI_LONG_LONG, MPI_MAX, comm);

    //     if (max_size_ll < 0) {
    //         if (rank == root) std::cerr << "Invalid max_size computed.\n";
    //         MPI_Abort(comm, 1);
    //     }

    //     // MPI collectives here still take int counts.
    //     if (max_size_ll > static_cast<long long>(std::numeric_limits<int>::max())) {
    //         if (rank == root) {
    //             std::cerr
    //                 << "Error: max serialized size (" << max_size_ll
    //                 << ") exceeds INT_MAX, cannot use MPI_Allgather with int counts.\n"
    //                 << "Need chunking or large-count collectives (MPI-4 *_c variants) if your MPI supports them.\n";
    //         }
    //         MPI_Abort(comm, 2);
    //     }

    //     const int max_size = static_cast<int>(max_size_ll);

    //     // We also need the true sizes on root to know how many bytes to deserialize.
    //     // This is safe because each local message size <= max_size <= INT_MAX.
    //     int local_size_int = static_cast<int>(serialized.size());
    //     std::vector<int> recv_sizes;
    //     if (rank == root) recv_sizes.resize(comm_size);

    //     MPI_Gather(&local_size_int, 1, MPI_INT,
    //             rank == root ? recv_sizes.data() : nullptr, 1, MPI_INT,
    //             root, comm);

    //     // 3) Pad serialized data to max_size and allgather fixed-size blocks
    //     std::vector<char> sendbuf(static_cast<size_t>(max_size), '\0');
    //     if (!serialized.empty()) {
    //         std::memcpy(sendbuf.data(), serialized.data(), static_cast<size_t>(local_size_int));
    //     }

    //     // All ranks receive (comm_size * max_size) bytes (since you requested Allgather).
    //     // If you only need root to receive, MPI_Gather would be much cheaper.
    //     std::vector<char> recvbuf(static_cast<size_t>(comm_size) * static_cast<size_t>(max_size));
    //     MPI_Allgather(sendbuf.data(), max_size, MPI_CHAR,
    //                 recvbuf.data(), max_size, MPI_CHAR,
    //                 comm);

    //     // 4) Deserialize on root using the true sizes
    //     if (rank == root) {
    //         all_buffer.clear();
    //         all_buffer.resize(comm_size);

    //         for (int i = 0; i < comm_size; i++) {
    //             const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(max_size);
    //             const int true_n = recv_sizes[i];

    //             // true_n bytes from that rank’s padded block
    //             std::string s(recvbuf.data() + offset, recvbuf.data() + offset + static_cast<size_t>(true_n));
    //             all_buffer[i] = deserializeData<std::vector<T>>(s);
    //         }
    //     }
    // }

    // template <typename T>
    // void gatherVectorToRoot(std::vector<T> &data,
    //                         std::vector<std::vector<T>> &all_buffer,
    //                         int root = 0,
    //                         MPI_Comm comm = MPI_COMM_WORLD)
    // {
    //     int rank, size;
    //     MPI_Comm_rank(comm, &rank);
    //     MPI_Comm_size(comm, &size);

    //     // Serialize local vector
    //     std::string serialized = serializeData(data);
    //     int send_size = static_cast<int>(serialized.size());

    //     std::cout << "Rank " << rank << " - send size = " << send_size << "; serialized size = " << serialized.size() << std::endl;
    //     // 1. Gather sizes on root
    //     std::vector<int> recv_sizes;
    //     if (rank == root) recv_sizes.resize(size);

    //     std::cout << "Rank " << rank << " - XA" << std::endl;
    //     MPI_Gather(&send_size, 1, MPI_INT,
    //             recv_sizes.data(), 1, MPI_INT,
    //             root, comm);

    //     std::cout << "Rank " << rank << " - XB" << std::endl;

    //     // 2. Compute displacements + total size (root only)
    //     std::vector<long> displs;
    //     std::vector<char> recvbuf;
    //     long long total_size = 0;
    //     long long expected_size = 0;
        
    //     if (rank == root) {
    //         displs.resize(size);
    //         for (int i = 0; i < size; i++) {
    //             displs[i] = total_size;
    //             total_size += recv_sizes[i];
    //             expected_size += recv_sizes[i];
    //         }
    //         std::cout << "Rank " << rank << " - total size = " << total_size << "; expected size = " << expected_size << std::endl;
    //         recvbuf.resize(total_size);
    //         std::cout << "Rank " << rank << " - XC" << std::endl;

            
    //     }

    //     // 3. Gather variable-sized serialized buffers
    //     MPI_Gatherv(serialized.data(), send_size, MPI_CHAR,
    //                 recvbuf.data(), recv_sizes.data(), displs.data(), MPI_CHAR,
    //                 root, comm);

    //     // 4. Deserialize on root
    //     if (rank == root) {
    //         all_buffer.clear();
    //         all_buffer.resize(size);

    //         for (int i = 0; i < size; i++) {
    //             std::string s(recvbuf.data() + displs[i], recv_sizes[i]);
    //             all_buffer[i] = deserializeData<std::vector<T>>(s);
    //         }
    //     }
    // }

    template <typename T>
    void gatherToRoot(
        T &data,
        std::vector<T> &all_buffer,
        int root = 0,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        int size = all_buffer.size();

        // Serialize
        std::string serialized = serializeData(data);
        int send_size = serialized.size();

        // Root allocates receive buffer
        std::vector<char> recvbuf;
        char* recvptr = nullptr;

        if (rank == root) {
            recvbuf.resize(size * send_size);
            recvptr = recvbuf.data();
        }

        // Gather
        MPI_Gather(
            serialized.data(), send_size, MPI_CHAR,
            recvptr,           send_size, MPI_CHAR,
            root, comm
        );

        // Only root deserializes
        if (rank == root) {
            for (int i = 0; i < size; i++) {
                std::string s(&recvbuf[i * send_size], send_size);
                all_buffer[i] = deserializeData<T>(s);
            }
        }
    }

    template <typename T>
    void gatherHalfToRoot(
        const T &local_data,
        std::vector<T> &row_buffer,   // root will store full row here
        int row_index,                // row index in the full matrix
        int root = 0,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Determine if this rank participates (upper-half only)
        bool has_data = (rank >= row_index);

        // Serialize only if this rank has data
        std::string serialized;
        int send_size = 0;
        if (has_data) {
            serialized = serializeData(local_data);
            send_size = serialized.size();
        }

        // Gather send_sizes to root
        std::vector<int> send_sizes(size, 0);
        MPI_Gather(&send_size, 1, MPI_INT,
                send_sizes.data(), 1, MPI_INT,
                root, comm);

        // Compute offsets (displacements) in the full row buffer
        std::vector<int> displs(size, 0);
        int total_recv_size = 0;
        if (rank == root) {
            for (int i = 0; i < size; i++) {
                if (send_sizes[i] > 0) {
                    displs[i] = total_recv_size;
                    total_recv_size += send_sizes[i];
                }
            }
        }

        // Allocate receive buffer on root
        std::vector<char> recvbuf(total_recv_size);
        char* recvptr = rank == root ? recvbuf.data() : nullptr;

        // Perform Gatherv: only upper-half ranks send data
        MPI_Gatherv(
            has_data ? serialized.data() : nullptr, send_size, MPI_CHAR,
            recvptr, send_sizes.data(), displs.data(), MPI_CHAR,
            root, comm
        );

        // Root deserializes into the full row buffer
        if (rank == root) {
            // Ensure the row buffer is full size
            row_buffer.resize(size);

            for (int i = 0; i < size; i++) {
                if (send_sizes[i] > 0) {
                    std::string s(&recvbuf[displs[i]], send_sizes[i]);
                    // Place at the correct column (right-half)
                    row_buffer[i] = deserializeData<T>(s);
                }
            }
        }
    }


    template <typename T>
    void broadcastFromRoot(
        T &data,
        int root = 0,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        // Serialize on root
        std::string serialized;
        if (rank == root) {
            serialized = serializeData(data);
        }

        // First broadcast the size
        int send_size = serialized.size();
        MPI_Bcast(&send_size, 1, MPI_INT, root, comm);

        // Prepare buffer for all ranks
        if (rank != root) {
            serialized.resize(send_size);
        }

        // Broadcast serialized payload
        MPI_Bcast(serialized.data(), send_size, MPI_CHAR, root, comm);

        // Non-root ranks deserialize into "data"
        if (rank != root) {
            data = deserializeData<T>(serialized);
        }
    }

    template <typename T>
    void broadcastVectorFromRoot(std::vector<T> &data,
                                int root = 0,
                                MPI_Comm comm = MPI_COMM_WORLD)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);

        // Serialize vector only on root
        std::string serialized;
        int data_size = 0;
        if (rank == root) {
            serialized = serializeData(data);
            data_size = serialized.size();
        }

        // Broadcast the size first
        MPI_Bcast(&data_size, 1, MPI_INT, root, comm);

        // Resize buffer on non-root processes
        std::vector<char> buffer(data_size);
        if (rank == root) {
            std::copy(serialized.begin(), serialized.end(), buffer.begin());
        }

        // Broadcast the serialized data
        MPI_Bcast(buffer.data(), data_size, MPI_CHAR, root, comm);

        // Deserialize on non-root processes
        if (rank != root) {
            std::string s(buffer.begin(), buffer.end());
            data = deserializeData<std::vector<T>>(s);
        }
    }


    template <typename T>
    void sendData(
        const T &data,
        int destRank,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        int rank;
        MPI_Comm_rank(comm, &rank);

        // --- 1. Serialize ---
        std::string serialized = serializeData(data);
        int send_size = serialized.size();

        // --- 2. Send size ---
        MPI_Send(&send_size, 1, MPI_INT, destRank, 0, comm);

        // --- 3. Send actual data ---
        MPI_Send(serialized.data(), send_size, MPI_CHAR, destRank, 0, comm);
    }



    template <typename T>
    void recvData(
        T &data,
        int srcRank,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        MPI_Status status;

        // --- 1. Receive the size first ---
        int recv_size = 0;
        MPI_Recv(&recv_size, 1, MPI_INT, srcRank, 0, comm, &status);

        // --- 2. Allocate buffer and receive payload ---
        std::string buffer;
        buffer.resize(recv_size);

        MPI_Recv(buffer.data(), recv_size, MPI_CHAR, srcRank, 0, comm, &status);

        // --- 3. Deserialize ---
        data = deserializeData<T>(buffer);
    }


    template <typename T>
    void gatherUpperTriangleRowToLeft(
        T &data,
        std::vector<T> &rowData,
        int col,
        int row,
        int rowSize,
        MPI_Comm row_comm
    ) {
        const int ROOT = 0;          // leftmost column in row_comm
        const int TAG_BASE = 5000;   // clean namespace for MPI tags

        int rank;
        MPI_Comm_rank(row_comm, &rank);
        bool isRoot = (col == ROOT);

        if (isRoot) {
            rowData.resize(rowSize);
        }

        MPI_Status status;

        // Loop over all columns j in this row
        for (int j = 0; j < rowSize; j++) {

            if(row < col) { // upper half
            
            }
            else if(row > col) { //lower half

            }
            
        }
    }


}


