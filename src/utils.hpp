#pragma once

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "colors.h"

using FLOAT = double;

static std::vector<std::string> split(const std::string& s, char seperator) {
  std::vector<std::string> tokens;
  std::string token;
  std::stringstream ss(s);
  while (std::getline(ss, token, seperator)) {
    tokens.push_back(token);
  }
  return tokens;
}

template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  for (auto v : vec) {
    os << v << " ";
  }
  return os;
}

class RndNormal {
private:
  std::mt19937* gen;
  std::normal_distribution<double>* dist;

public:
  RndNormal() {
    auto seed = std::random_device{}();
#ifdef FIXED_RANDOM_SEED
    seed = 1172407366; //
#endif
    gen = new std::mt19937(seed);
    dist = new std::normal_distribution<double>(0, 1);
    // std::cout << "Initialized random generator (normal distribution with mean 0 and stddev 1) and seed " << seed << "\n";
  }

  ~RndNormal() {
    delete gen;
    delete dist;
  }

  inline FLOAT get() { return (*dist)(*gen); }
};

class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> t;
  std::string name;
  bool done;

public:
  Timer(const std::string& name) {
    done = false;
    t = std::chrono::high_resolution_clock::now();
    this->name = name;
  }

  void stop() {
    done = true;
    std::chrono::duration<double> t_elapsed = std::chrono::high_resolution_clock::now() - t;
    t_elapsed = std::chrono::high_resolution_clock::now() - t;
    std::cout << KLCYAN << "Timing: " << name << " took " << std::setprecision(2) << std::fixed << t_elapsed.count() << " s"
              << KNORM << std::endl;
  }

  ~Timer() {
    if (done) { return; }
    std::chrono::duration<double> t_elapsed = std::chrono::high_resolution_clock::now() - t;
    t_elapsed = std::chrono::high_resolution_clock::now() - t;
    std::cout << KLCYAN << "Timing: " << name << " took " << std::setprecision(2) << std::fixed << t_elapsed.count() << " s"
              << KNORM << std::endl;
  }
};

class IntermittentTimer {
  std::chrono::time_point<std::chrono::high_resolution_clock> t;
  double t_elapsed, t_interval;
  std::string name, interval_label;

public:
  IntermittentTimer(const std::string& name, std::string interval_label = "") : name{name}, interval_label{interval_label} {
    t_elapsed = 0;
    t_interval = 0;
  }

  void done() {
    std::cout << KLCYAN << "Timing: " << name << " took " << std::setprecision(2) << std::fixed << t_elapsed << " s" << KNORM
              << "\n";
  }

  std::string get_final() {
    std::ostringstream os;
    os << name << ": " << std::setprecision(2) << std::fixed << t_elapsed;
    return os.str();
  }

  void start() {
    if (!interval_label.empty()) { std::cout << std::left << std::setw(40) << interval_label + ":" << KNORM << "\n"; }
    t = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    std::chrono::duration<double> interval = std::chrono::high_resolution_clock::now() - t;
    t_interval = interval.count();
    t_elapsed += t_interval;
    if (!interval_label.empty()) {
      std::cout << KBLUE << std::setprecision(2) << std::fixed << t_interval << " s" << KNORM << "\n";
    }
  }

  double get_interval() { return t_interval; }
};

class NullBuffer : public std::streambuf {
public:
  int overflow(int c) { return c; }
};

static std::string perc_str(int64_t num, int64_t tot) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  os << num << " (" << (tot == 0 ? 0.0 : (100.0 * num / tot)) << "%)";
  return os.str();
}
