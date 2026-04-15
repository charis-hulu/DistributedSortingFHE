#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;


std::vector<double> loadPoints1D(
    const size_t vectorLength
)
{
    std::vector<double> v(vectorLength, 0.0);
    std::ifstream file("data/ext_points1d.csv");

    if (!file.is_open())
    {
        file.open("../data/ext_points1d.csv");
        if (!file.is_open())
            std::cerr << "Error: Could not open the file!" << std::endl;
    }

    std::string line;
    size_t count = 0;

    while (std::getline(file, line) && count < vectorLength)
    {
        try
        {
            v[count] = std::stod(line);
            count++;
        } catch (const std::exception& e)
        {
            std::cerr << "Error: Invalid number format in the file." << std::endl;
        }
    }

    file.close();

    return v;
}

int main() {
    std::string caseName;
    size_t vectorLength, subVectorLength;
    // int a, b;

    // ---- User Input ----
    std::cout << "Enter test case name (e.g., case_3): ";
    std::cin >> caseName;

    std::cout << "Enter vectorLength: ";
    std::cin >> vectorLength;

    std::cout << "Enter subVectorLength: ";
    std::cin >> subVectorLength;

    // std::cout << "Enter min value (a): ";
    // std::cin >> a;

    // std::cout << "Enter max value (b): ";
    // std::cin >> b;

    // ---- Directory ----
    std::string dir = "/eagle/dist_relational_alg/chulu/fhe_test_cases/" + caseName;
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
        std::cout << "Created directory: " << dir << "\n";
    }

    std::string outputFile = dir + "/input.json";

    // ---- Generate vector ----
    std::vector<double> v = loadPoints1D(vectorLength);
    
    // ---- JSON ----
    json j;
    

    j["subVectorLength"] = subVectorLength;
    j["dg_c"] = 3;
    j["df_c"] = 2;
    j["dg_i"] = (log2(vectorLength) + 1) / 2;
    j["df_i"] = 2;
    j["integralPrecision"] = 1;
    j["decimalPrecision"] = 59;
    j["enableBootstrap"] = false;
    j["ringDim"] = 0;
    j["verbose"] = true;
    j["v"] = v;

    // ---- Save file ----
    std::ofstream ofs(outputFile);
    ofs << j.dump(4);
    ofs.close();

    std::cout << "Generated: " << outputFile << "\n";
    std::cout << "Done.\n";
    return 0;
}
