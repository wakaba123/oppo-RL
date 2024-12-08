#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sstream>   // For std::istringstream
#include <string>    // For std::string
#include <iostream>  // For std::cout, std::endl
#include <arpa/inet.h>

void reset_stats(const std::vector<std::string>& policies) {
    for (const auto& policy : policies) {
        std::string path = "/sys/devices/system/cpu/cpufreq/" + policy + "/stats/reset";
        FILE* file = fopen(path.c_str(), "w");
        if (file) {
            fprintf(file, "1");
            fclose(file);
        } else {
            std::cerr << "Failed to reset stats for " << policy << "\n";
        }
    }
}

std::vector<std::string> get_time_in_state(const std::vector<std::string>& policies) {
    std::vector<std::string> outputs;
    for (const auto& policy : policies) {
        std::string path = "/sys/devices/system/cpu/cpufreq/" + policy + "/stats/time_in_state";
        FILE* file = fopen(path.c_str(), "r");
        std::string output;
        if (file) {
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), file) != nullptr) {
                output += buffer;
            }
            fclose(file);
            outputs.push_back(output);
        } else {
            std::cerr << "Failed to read time_in_state for " << policy << "\n";
        }
    }
    return outputs;
}

double calculate_power(const std::array<std::vector<int>, 4>& freqs,
                       const std::array<std::vector<double>, 4>& powers,
                       const std::vector<std::string>& time_in_state1,
                       const std::vector<std::string>& time_in_state2) {
    double total_power = 0;
    std::array<int, 4> cpu_nums = {2, 3, 2, 1};

    for (size_t i = 0; i < time_in_state1.size(); ++i) {
        std::istringstream iss(time_in_state1[i]);
        std::string line;
        std::istringstream iss2(time_in_state2[i]);
        std::string line2;

        while (std::getline(iss, line) && std::getline(iss2, line2)) {
            int freq, freq_time;
            std::istringstream linestream(line);
            linestream >> freq >> freq_time;

            std::istringstream linestream2(line2);
            int freq2, freq_time2;
            linestream2 >> freq2 >> freq_time2;

            if (freq != freq2) {
                printf("here error ,freq not equals to freq2\n");
            }

            auto it = std::find(freqs[i].begin(), freqs[i].end(), freq);
            if (it != freqs[i].end()) {
                size_t idx = std::distance(freqs[i].begin(), it);
                total_power += ((freq_time2 - freq_time) / 100.0) * powers[i][idx] * cpu_nums[i];
            }
        }
    }
    return total_power;
}

double calculate_time(
    const std::vector<std::string>& time_in_state1,
    const std::vector<std::string>& time_in_state2) {
    for (size_t i = 0; i < time_in_state1.size(); ++i) {
        std::istringstream iss(time_in_state1[i]);
        std::string line;
        std::istringstream iss2(time_in_state2[i]);
        std::string line2;
        std::cout << "cluster " << i << std::endl;
        std::ofstream outfile("cluster_" + std::to_string(i) + ".txt");
        if (!outfile.is_open()) {
            std::cerr << "无法打开文件。" << std::endl;
            return 1;  // 返回错误代码
        }

        while (std::getline(iss, line) && std::getline(iss2, line2)) {
            int freq, freq_time;
            std::istringstream linestream(line);
            linestream >> freq >> freq_time;

            std::istringstream linestream2(line2);
            int freq2, freq_time2;
            linestream2 >> freq2 >> freq_time2;

            if (freq != freq2) {
                printf("here error ,freq not equals to freq2\n");
            }

            std::cout << freq << "," << freq_time2 - freq_time << std::endl;
            outfile << freq << "," << freq_time2 - freq_time << std::endl;
        }
        outfile.close();
    }
    return 0;
}

int main() {
    std::vector<std::string> policies = {"policy0", "policy2", "policy5", "policy7"};
    std::array<std::vector<int>, 4> freqs = {
        std::vector<int>{364800, 460800, 556800, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1344000, 1459200, 1574400, 1689600, 1804800, 1920000, 2035200, 2150400, 2265600},
        std::vector<int>{499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800, 3014400, 3072000, 3148800},
        std::vector<int>{499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800},
        std::vector<int>{480000, 576000, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1363200, 1478400, 1593600, 1708800, 1824000, 1939200, 2035200, 2112000, 2169600, 2246400, 2304000, 2380800, 2438400, 2496000, 2553600, 2630400, 2688000, 2745600, 2803200, 2880000, 2937600, 2995200, 3052800}};
    std::array<std::vector<double>, 4> powers = {
        std::vector<double>{4, 5.184, 6.841, 8.683, 10.848, 12.838, 14.705, 17.13, 19.879, 21.997, 25.268, 28.916, 34.757, 40.834, 46.752, 50.616, 56.72, 63.552},
        std::vector<double>{15.386, 19.438, 24.217, 28.646, 34.136, 41.231, 47.841, 54.705, 58.924, 68.706, 77.116, 86.37, 90.85, 107.786, 121.319, 134.071, 154.156, 158.732, 161.35, 170.445, 183.755, 195.154, 206.691, 217.975, 235.895, 245.118, 258.857, 268.685, 289.715, 311.594, 336.845, 363.661},
        std::vector<double>{15.53, 20.011, 24.855, 30.096, 35.859, 43.727, 51.055, 54.91, 64.75, 72.486, 80.577, 88.503, 99.951, 109.706, 114.645, 134.716, 154.972, 160.212, 164.4, 167.938, 178.369, 187.387, 198.433, 209.545, 226.371, 237.658, 261.999, 275.571, 296.108},
        std::vector<double>{31.094, 39.464, 47.237, 59.888, 70.273, 84.301, 97.431, 114.131, 126.161, 142.978, 160.705, 181.76, 201.626, 223.487, 240.979, 253.072, 279.625, 297.204, 343.298, 356.07, 369.488, 393.457, 408.885, 425.683, 456.57, 481.387, 511.25, 553.637, 592.179, 605.915, 655.484}};

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    const int PORT = 8080;
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // 绑定 socket 到端口
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    std::vector<std::string> time_in_state1, time_in_state2;

    int t = 0;
    int flag = 0;
    while (true) {
        // std::cout << "Waiting for the first socket message..." << std::endl;

        // 接收第一个 socket 包
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("Accept");
            close(server_fd);
            exit(EXIT_FAILURE);
        }
        // std::cout << "here get first package" << std::endl;

        char buffer[1024] = {0};
        read(new_socket, buffer, 1024);
        flag = atoi(buffer);
        // std::cout << "here flag is " << buffer<< std::endl;

        if (flag == 0) {
            time_in_state1 = get_time_in_state(policies);  // 第一次读取 time_in_state
        } else if (flag == 1) {
            // std::cout << "here flag is 1, print power during this time" << std::endl;
            time_in_state2 = get_time_in_state(policies);  // 第二次读取 time_in_state
            double power = calculate_power(freqs, powers, time_in_state1, time_in_state2);
            std::cout << power << std::endl;
            std::string data = std::to_string(power);
            send(new_socket, data.c_str(), data.length(), 0);
        } else if (flag == 2) {
            // std::cout << "here flag is 2, print time_in_state" << std::endl;
            time_in_state2 = get_time_in_state(policies);  // 第二次读取 time_in_state
            calculate_time(time_in_state1, time_in_state2);
        }

        close(new_socket);
    }

    close(server_fd);
    return 0;
}
