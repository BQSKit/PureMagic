#include <assert.h>
#include <deque>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>

using namespace std;

void signal_handler(int signal) {
  void* array[10];
  size_t size;

  // Get void*'s for all entries currently on the stack
  size = backtrace(array, 100);

  // Print out all the frames to stderr
  std::cerr << "Error: signal " << signal << "\n";
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

static vector<string> split_string(const string& s, char delim) {
  vector<string> elems;
  stringstream ss(s);
  string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

struct PairHash {
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

ostream& operator<<(ostream& os, const vector<int>& v) {
  os << "{";
  for (int i = 0; i < v.size(); i++) {
    os << v[i];
    if (i < v.size() - 1) { os << ", "; }
  }
  os << "}";
  return os;
}

static bool basis_commutes_with(int b1, int b2) {
  return (b1 == 'I' || b2 == 'I' || b1 == b2);
}

static int add_phase(int phase1, int phase2) {
  return (phase1 + phase2) % 4;
}

const static vector<string> PHASES = {"+1", "+i", "-1", "-i"};
const static unordered_map<pair<char, char>, pair<char, int>, PairHash> TERM_MUL = {                     //
        {{'I', 'I'}, {'I', 0}}, {{'I', 'X'}, {'X', 0}}, {{'I', 'Y'}, {'Y', 0}}, {{'I', 'Z'}, {'Z', 0}},  // I
        {{'X', 'X'}, {'I', 0}}, {{'X', 'I'}, {'X', 0}}, {{'X', 'Y'}, {'Z', 1}}, {{'X', 'Z'}, {'Y', 3}},  // X
        {{'Y', 'Y'}, {'I', 0}}, {{'Y', 'X'}, {'Z', 3}}, {{'Y', 'I'}, {'Y', 0}}, {{'Y', 'Z'}, {'X', 1}},  // Y
        {{'Z', 'Z'}, {'I', 0}}, {{'Z', 'I'}, {'Z', 0}}, {{'Z', 'X'}, {'Y', 1}}, {{'Z', 'Y'}, {'X', 3}}}; // Z
class PauliTerm {

public:
  char basis;
  int phase;
  int qubit;

  void load_from_string(const string& s) {
    try {
      if (!s.starts_with("Pauli")) {
        throw runtime_error(to_string(__LINE__) + ": error loading Pauli product term from string " + s);
      }
      basis = s[5];
      assert(basis == 'X' || basis == 'Y' || basis == 'Z');
      string phase_str = s.substr(7, 2);
      for (int i = 0; i < PHASES.size(); i++) {
        if (phase_str == PHASES[i]) {
          phase = i;
          break;
        }
      }
      qubit = stoi(s.substr(10));
    } catch (const exception& ex) {
      cerr << __LINE__ << "Exception: " << ex.what() << " s " << s << " " << s.substr(7, 2) << "\n";
      throw ex;
    }
  }

  friend ostream& operator<<(ostream& os, const PauliTerm& term) {
    os << "Pauli" << term.basis << "(" << PHASES[term.phase] << ")" << term.qubit;
    return os;
  }

  bool operator<(const PauliTerm& other) const { return qubit < other.qubit; }

  PauliTerm& operator=(const PauliTerm& other) {
    basis = other.basis;
    phase = other.phase;
    qubit = other.qubit;
    return *this;
  }

  void set_term(char basis, int phase2) {
    basis = basis;
    phase = add_phase(phase, phase2);
  }

  PauliTerm commute_right(PauliTerm* rhs, int angle_numerator, int angle_denominator) {
    if (qubit != rhs->qubit || basis_commutes_with(basis, rhs->basis)) { return *rhs; }
    PauliTerm new_term = {.basis = 'I', .phase = add_phase(phase, rhs->phase), .qubit = qubit};
    auto [new_basis, phase_shift] = TERM_MUL.at({basis, rhs->basis});
    new_term.basis = new_basis;
    new_term.phase = add_phase(new_term.phase, phase_shift);
    new_term.phase = add_phase(new_term.phase, 1);
    if (angle_numerator > angle_denominator) { new_term.phase = add_phase(new_term.phase, 2); }
    return new_term;
  }
};

class PauliProduct {
private:
  vector<PauliTerm> terms;
  int angle_numerator = 1;
  int angle_denominator = 1;

public:
  bool is_clifford;

  void load_from_string(const string& s) {
    auto tokens = split_string(s, '<');
    if (tokens.size() != 2 || !tokens[1].starts_with("Angle")) {
      throw runtime_error(to_string(__LINE__) + ": Error loading Pauli product angle from string " + s);
    }
    auto angle_string = tokens[1].substr(string("Angle(").length());
    auto factors = split_string(angle_string, '/');
    if (factors[0].length() > 2) { angle_numerator = stoi(factors[0].substr(0, factors[0].length() - 2)); }
    if (factors.size() == 2) { angle_denominator = stoi(factors[1].substr(0, factors[1].length() - 2)); }
    // simplify angle to determine if is clifford
    double gcd_factor = gcd(angle_numerator, angle_denominator);
    int numerator = floor(angle_numerator / gcd_factor);
    int denominator = floor(angle_denominator / gcd_factor);
    is_clifford = denominator == -4 || denominator == -2 || denominator == -1 || denominator == 0 || denominator == 1 ||
                  denominator == 2 || denominator == 4;

    auto term_tokens = split_string(tokens[0], '.');
    for (auto& token : term_tokens) {
      PauliTerm term;
      term.load_from_string(token);
      terms.push_back(term);
    }
  }

  friend ostream& operator<<(ostream& os, const PauliProduct& pp) {
    os << pp.terms[0];
    for (int i = 1; i < pp.terms.size(); i++) {
      os << "." << pp.terms[i];
    }
    if (pp.angle_denominator == 1) {
      os << "<Angle(" << pp.angle_numerator << "pi)>";
    } else if (pp.angle_numerator == 1) {
      os << "<Angle(pi/" << pp.angle_denominator << ")>";
    } else {
      os << "<Angle(" << pp.angle_numerator << "pi/" << pp.angle_denominator << ")>";
    }
    return os;
  }

  bool commutes_with(PauliProduct& other) {
    vector<pair<PauliTerm, PauliTerm>> shared_terms;
    unordered_map<int, char> terms_map;
    for (auto& term : terms) {
      terms_map.insert({term.qubit, term.basis});
    }
    int sum_signs = 0;
    for (auto& term : other.terms) {
      auto res = terms_map.find(term.qubit);
      if (res != terms_map.end()) {
        if (!basis_commutes_with(term.basis, res->second)) { sum_signs++; }
      }
    }
    // cout << "Sum signs " << sum_signs << " commutes with " << (sum_signs % 2 == 0) << "\n";
    return (sum_signs % 2 == 0);
  }

  PauliProduct commute_right(PauliProduct& rhs) {
    if (commutes_with(rhs)) { return rhs; }
    //  Ensure we're commuting a Clifford angle rotation
    if (!is_clifford) { throw runtime_error("Currently only support commuting right of Clifford angles"); }
    unordered_map<int, pair<PauliTerm*, PauliTerm*>> all_terms_map;
    for (auto& term : terms) {
      all_terms_map.insert({term.qubit, {&term, nullptr}});
    }
    for (auto& term : rhs.terms) {
      auto it = all_terms_map.find(term.qubit);
      if (it == all_terms_map.end()) {
        all_terms_map.insert({term.qubit, {nullptr, &term}});
      } else {
        it->second.second = &term;
      }
    }
    PauliProduct new_prod;
    new_prod.terms.resize(all_terms_map.size());
    new_prod.angle_denominator = rhs.angle_denominator;
    new_prod.angle_numerator = rhs.angle_numerator;
    int i = 0;
    for (const auto& [qubit, term_pair] : all_terms_map) {
      if (term_pair.second == nullptr) {
        new_prod.terms[i] = *term_pair.first;
      } else {
        if (term_pair.first != nullptr) {
          new_prod.terms[i] = term_pair.first->commute_right(term_pair.second, angle_numerator, angle_denominator);
        } else {
          new_prod.terms[i] = *term_pair.second;
        }
      }
      i++;
    }
    return new_prod;
  }
};

class PauliProductDAGNode {
private:
  vector<int> parse_id_list(const string& s) {
    if (s == "set()") { return {}; }
    vector<int> ids;
    auto tokens = split_string(s.substr(1), ',');
    for (auto& token : tokens) {
      try {
        ids.push_back(stoi(token));
      } catch (const exception& ex) {
        cerr << __LINE__ << ": Exception: " << ex.what() << " s " << s << " " << token << "\n";
        throw ex;
      }
    }
    return ids;
  }

public:
  int id;
  PauliProduct product;
  vector<int> children;
  vector<int> parents;

  void load_from_string(string& s) {
    const int NUM_SPACE_TOKENS = 4;
    s = regex_replace(s, regex(", "), ",");
    auto tokens = split_string(s, ' ');
    if (tokens.size() != NUM_SPACE_TOKENS) {
      throw runtime_error(to_string(__LINE__) + ": Incorrect number of tokens: expected " + to_string(NUM_SPACE_TOKENS) +
                          " but got " + to_string(tokens.size()) + " for line:\n" + s);
    }
    try {
      id = stoi(tokens[0]);
    } catch (const exception& ex) {
      cerr << __LINE__ << ": Exception: " << ex.what() << " s " << s << " " << tokens[0] << "\n";
      throw ex;
    }
    product.load_from_string(tokens[1]);
    children = parse_id_list(tokens[2]);
    parents = parse_id_list(tokens[3]);
  }

  friend ostream& operator<<(ostream& os, const PauliProductDAGNode& node) {
    os << node.id << " " << node.product << " ";
    if (node.children.empty()) {
      os << "set() ";
    } else {
      os << node.children << " ";
    }
    if (node.parents.empty()) {
      os << "set()";
    } else {
      os << node.parents;
    }
    return os;
  }

  bool is_root() { return parents.empty(); }

  bool is_clifford() { return product.is_clifford; }
};

class PauliProductDAG {
private:
  vector<PauliProductDAGNode> nodes;
  unordered_set<int> roots;
  vector<int> topological_order;

public:
  void load_from_file(const string& fname) {
    cout << "Loading circuit from " << fname << "\n";
    ifstream f(fname);
    if (!f) { throw runtime_error("Could not open " + fname + "\n"); }
    vector<string> lines;
    string buf;
    while (getline(f, buf)) {
      lines.push_back(buf);
    }
    cout << "Found " << lines.size() << " lines in " << fname << "\n";
    nodes.resize(lines.size());
    topological_order.resize(lines.size());
    int num_cliffords = 0;
    for (int i = 0; i < lines.size(); i++) {
      auto& node = nodes[i];
      node.load_from_string(lines[i]);
      assert(node.id == i);
      if (node.is_root()) { roots.insert(node.id); }
      if (node.is_clifford()) { num_cliffords++; }
      topological_order[node.id] = i;
    }
    cout << "Loaded " << nodes.size() << " products from " << fname << " of which " << num_cliffords << " are cliffords and "
         << roots.size() << " are roots\n";
  }

  friend ostream& operator<<(ostream& os, const PauliProductDAG& dag) {
    for (auto& node : dag.nodes) {
      os << node << "\n";
    }
    return os;
  }

  bool is_uncommuted_nonclifford(int node_id, set<int>& uncommuted_noncliffords) {
    if (nodes[node_id].is_clifford()) { return false; }
    if (uncommuted_noncliffords.contains(node_id)) { return true; }
    for (auto child_id : nodes[node_id].children) {
      if (nodes[child_id].is_clifford() || is_uncommuted_nonclifford(child_id, uncommuted_noncliffords)) {
        uncommuted_noncliffords.insert(node_id);
        return true;
      }
    }
    return false;
  }

  bool done_commuting_nonclifford(int node_id, set<int>& uncommuted_noncliffords) {
    for (auto parent_id : nodes[node_id].parents) {
      if (nodes[parent_id].is_clifford() || uncommuted_noncliffords.contains(parent_id)) { return false; }
    }
    return true;
  }

  bool indirect_path_exists(int u, int v) {
    if (u == v) { return true; }
    int topo_index_u = topological_order[u];
    int topo_index_v = topological_order[v];
    if (topo_index_u > topo_index_v) { return false; }
    // BFS traversal from u's children that are not v
    deque<int> q;
    unordered_set<int> visited;
    for (auto child_id : nodes[u].children) {
      if (child_id != v) {
        q.push_back(child_id);
        visited.insert(child_id);
      }
    }
    if (q.empty()) { return false; }
    while (!q.empty()) {
      int node_id = q.front();
      q.pop_front();
      for (auto child_id : nodes[node_id].children) {
        if (child_id == v) { return true; }
        if (!visited.contains(child_id)) {
          // Prune if v cannot depend on child_id
          if (topological_order[child_id] > topo_index_v) { continue; }
          visited.insert(child_id);
          q.push_back(child_id);
        }
      }
    }
    return false;
  }

  vector<int> get_valid_parent_cliffords(int node_id) {
    vector<int> parent_cliffords;
    for (auto parent_id : nodes[node_id].parents) {
      if (nodes[parent_id].is_clifford() && !indirect_path_exists(parent_id, node_id)) { parent_cliffords.push_back(parent_id); }
    }
    return parent_cliffords;
  }

  int youngest_parent(vector<int>& related_nodes) {
    int youngest_id = 0;
    int min_val = numeric_limits<int>::max();
    for (auto node_id : related_nodes) {
      if (-topological_order[node_id] < min_val) {
        min_val = -topological_order[node_id];
        youngest_id = node_id;
      }
    }
    return youngest_id;
  }

  void update_product(int node_id, PauliProduct& pp) {}

  void swap_nodes(int n1, int n2) {}

  void update_topological_order_starting_at(int start_node_id) {}

  void commute_clifford_right(int clifford_id, int node_id) {
    if (!nodes[clifford_id].is_clifford()) { return; }
    auto& clifford = nodes[clifford_id];
    auto& node = nodes[node_id];
    PauliProduct new_node_prod = clifford.product.commute_right(node.product);
    cout << "  new product " << new_node_prod << " old product " << node.product << "\n";
    update_product(node_id, new_node_prod);
    swap_nodes(clifford_id, node_id);
    // If node or clifford has a higher topological order than any of their children, we must recompute the topological order
    for (auto child_id : node.children) {
      if (topological_order[child_id] > topological_order[node_id]) {
        update_topological_order_starting_at(node_id);
        return;
      }
    }
    for (auto child_id : clifford.children) {
      if (topological_order[child_id] > topological_order[clifford_id]) {
        update_topological_order_starting_at(node_id);
        return;
      }
    }
  }

  void commute_all_cliffords() {
    set<int> uncommuted_noncliffords;
    for (auto& node : nodes) {
      is_uncommuted_nonclifford(node.id, uncommuted_noncliffords);
    }
    ofstream f("uncommuted_noncliffords-mine");
    for (auto node_id : uncommuted_noncliffords) {
      f << node_id << "\n";
    }
    f.close();
    if (uncommuted_noncliffords.empty()) {
      cout << "No uncommuted non-Cliffords\n";
      return;
    }

    cout << "Commuting " << uncommuted_noncliffords.size() << " uncommuted noncliffords\n";

    ofstream dbg_f("debug.txt");
    int num_commuted = 0;
    while (uncommuted_noncliffords.size() > 0) {
      unordered_set<int> finished_noncliffords;
      for (auto node_id : uncommuted_noncliffords) {
        auto& node = nodes[node_id];
        if (done_commuting_nonclifford(node_id, uncommuted_noncliffords)) {
          finished_noncliffords.insert(node_id);
          dbg_f << "Finished nonclifford " << node_id << "\n";
          continue;
        }
        vector<int> parent_cliffords = get_valid_parent_cliffords(node_id);
        dbg_f << "node_id " << node_id << " parents " << parent_cliffords.size() << "\n";
        if (parent_cliffords.empty()) { continue; }
        for (auto parent_id : parent_cliffords) {
          dbg_f << "  valid_parent_id " << parent_id << "\n";
        }
        if (parent_cliffords.size() > 1) { cout << "parents " << parent_cliffords.size() << "\n"; }
        // check for loops
        for (auto parent_id : node.parents) {
          for (auto child_id : node.children) {
            if (child_id == parent_id) { throw runtime_error(to_string(__LINE__) + ": loop detected"); }
          }
        }
        int parent_id = youngest_parent(parent_cliffords);
        cout << "Commute " << parent_id << " past " << node_id << " (" << parent_cliffords.size() << ")\n";
        commute_clifford_right(parent_id, node_id);
        cout << "parent: " << nodes[parent_id] << "\n";
        cout << "node: " << node << "\n";
      }
      for (auto nonclifford_id : finished_noncliffords) {
        uncommuted_noncliffords.erase(nonclifford_id);
      }
      num_commuted += finished_noncliffords.size();
      break;
    }
    dbg_f.close();
  }
};

int main(int argc, char* argv[]) {
  signal(SIGABRT, signal_handler);
  signal(SIGSEGV, signal_handler);
  PauliProductDAG dag;
  string fname(argv[1]);
  dag.load_from_file(argv[1]);
  ofstream f("dag-loaded.txt");
  f << dag;
  f.close();
  dag.commute_all_cliffords();
}
