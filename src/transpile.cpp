#include <assert.h>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <queue>
#include <regex>
#include <set>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define USE_COLORS

#include "utils.hpp"

using namespace std;

// #define DBGTRACE
static bool traceon = true;

#ifdef DBGTRACE
#define DBG(x)                                                                                                                   \
  if (traceon) { cout << x; }
#else
// #define DBG 0 && cout
#define DBG(x)
#endif

static IntermittentTimer update_topo_timer("update_topo");

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
  os << "[";
  for (int i = 0; i < v.size(); i++) {
    os << v[i];
    if (i < v.size() - 1) { os << ", "; }
  }
  os << "]";
  return os;
}

ostream& operator<<(ostream& os, const set<int>& data) {
  os << "[";
  int i = 0;
  for (auto x : data) {
    os << x;
    if (i < data.size() - 1) { os << ", "; }
    i++;
  }
  os << "]";
  return os;
}

ostream& operator<<(ostream& os, const unordered_set<int>& data) {
  os << "[";
  int i = 0;
  for (auto x : data) {
    os << x;
    if (i < data.size() - 1) { os << ", "; }
    i++;
  }
  os << "]";
  return os;
}

ostream& operator<<(ostream& os, const unordered_map<int, int>& data) {
  os << "{";
  int i = 0;
  for (auto [k, v] : data) {
    os << k << ": " << v;
    if (i < data.size() - 1) { os << ", "; }
    i++;
  }
  os << "}";
  return os;
}

template <class T>
static bool contains(vector<T> const& v, T const& x) {
  return find(v.begin(), v.end(), x) != v.end();
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

  // Assuming this PauliTerm is on the left, commute it to the right past another PauliTerm
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

public:
  vector<PauliTerm> terms;
  int angle_numerator = 1;
  int angle_denominator = 1;

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
    set_angle(floor(angle_numerator / gcd_factor), floor(angle_denominator / gcd_factor));
    auto term_tokens = split_string(tokens[0], '.');
    for (auto& token : term_tokens) {
      PauliTerm term;
      term.load_from_string(token);
      terms.push_back(term);
    }
  }

  void set_angle(int numerator, int denominator) {
    angle_numerator = numerator;
    angle_denominator = denominator;
    is_clifford = denominator == -4 || denominator == -2 || denominator == -1 || denominator == 0 || denominator == 1 ||
                  denominator == 2 || denominator == 4;
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
    return (sum_signs % 2 == 0);
  }

  // Commute this measurement to the right past another
  PauliProduct commute_right(PauliProduct& rhs) {
    if (commutes_with(rhs)) {
      DBG(*this << " commutes with " << rhs << "\n");
      return rhs;
    }
    //  Ensure we're commuting a Clifford angle rotation
    if (!is_clifford) { throw runtime_error("Currently only support commuting right of Clifford angles"); }
    map<int, pair<PauliTerm*, PauliTerm*>> all_terms_map;
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
    new_prod.set_angle(rhs.angle_numerator, rhs.angle_denominator);

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
  set<int> parse_id_list(const string& s) {
    if (s == "[]") { return {}; }
    set<int> ids;
    auto tokens = split_string(s.substr(1), ',');
    for (auto& token : tokens) {
      try {
        ids.insert(stoi(token));
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
  set<int> children;
  set<int> parents;

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
    os << node.id << " " << node.product << " " << node.children << " " << node.parents;
    return os;
  }

  bool is_root() { return parents.empty(); }

  bool is_clifford() { return product.is_clifford; }

  bool involves_qubit(int qubit) {
    for (auto& term : product.terms) {
      if (term.qubit == qubit) { return true; }
    }
    return false;
  }
};

class PauliProductDAG {
private:
  vector<PauliProductDAGNode> nodes;
  unordered_set<int> roots;
  vector<int> topological_order;
  int max_qubit = 0;

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
      for (auto& term : node.product.terms) {
        max_qubit = max(max_qubit, term.qubit);
      }
    }
    cout << "Loaded " << nodes.size() << " products from " << fname << " of which " << num_cliffords << " are cliffords and "
         << roots.size() << " are roots, with max qubit " << max_qubit << "\n";
  }

  friend ostream& operator<<(ostream& os, const PauliProductDAG& dag) {
    for (auto& node : dag.nodes) {
      os << node << "\n";
    }
    return os;
  }

  bool is_uncommuted_nonclifford(int node_id, set<int>& uncommuted_noncliffords) {
    if (nodes[node_id].is_clifford()) { return false; }
    if (nodes[node_id].is_root()) { return true; }
    if (uncommuted_noncliffords.contains(node_id)) { return true; }
    if (nodes[node_id].children.empty()) {
      uncommuted_noncliffords.insert(node_id);
      return true;
    }
    for (auto child_id : nodes[node_id].children) {
      if (nodes[child_id].is_clifford() || is_uncommuted_nonclifford(child_id, uncommuted_noncliffords)) {
        uncommuted_noncliffords.insert(node_id);
        return true;
      }
    }
    return false;
  }

  bool done_commuting_nonclifford(int node_id, set<int>& uncommuted_noncliffords) {
#ifdef DBGTRACE
    DBG("check for done for " << node_id << " parents " << nodes[node_id].parents << "\n");
    for (auto parent_id : nodes[node_id].parents) {
      DBG("  parent " << parent_id << " clifford " << (nodes[parent_id].is_clifford() ? "True" : "False") << " uncommuted "
                      << (uncommuted_noncliffords.contains(parent_id) ? "True" : "False") << "\n");
    }
#endif
    for (auto parent_id : nodes[node_id].parents) {
      if (nodes[parent_id].is_clifford() || uncommuted_noncliffords.contains(parent_id)) { return false; }
    }
    return true;
  }

  // Check if an indirect forward path exists from u -> ... -> v
  bool indirect_path_exists(int u, int v) {
    if (u == v) { return true; }
    int topo_index_u = topological_order[u];
    int topo_index_v = topological_order[v];
    if (topo_index_u > topo_index_v) { return false; }
    // BFS traversal from u's children that are not v
    queue<int> q;
    unordered_set<int> visited;
    for (auto child_id : nodes[u].children) {
      if (child_id != v) {
        q.push(child_id);
        visited.insert(child_id);
      }
    }
    if (q.empty()) {
      DBG("    no path from " << u << " to " << v << " children " << nodes[u].children << "\n");
    } else {
      DBG("    path from " << u << " to " << v << " visited " << visited << "\n");
    }
    while (!q.empty()) {
      int node_id = q.front();
      q.pop();
      for (auto child_id : nodes[node_id].children) {
        if (child_id == v) { return true; }
        if (!visited.contains(child_id)) {
          // Prune if v cannot depend on child_id
          if (topological_order[child_id] > topo_index_v) { continue; }
          visited.insert(child_id);
          q.push(child_id);
        }
      }
    }
    return false;
  }

  vector<int> get_valid_parent_cliffords(int node_id) {
    vector<int> parent_cliffords;
    for (auto parent_id : nodes[node_id].parents) {
      if (!nodes[parent_id].is_clifford()) { continue; }
      if (!indirect_path_exists(parent_id, node_id)) { parent_cliffords.push_back(parent_id); }
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

  int oldest_child(vector<int>& related_nodes) {
    int oldest_id = 0;
    int max_val = numeric_limits<int>::min();
    for (auto node_id : related_nodes) {
      if (-topological_order[node_id] > max_val) {
        max_val = -topological_order[node_id];
        oldest_id = node_id;
      }
    }
    return oldest_id;
  }

  unordered_map<int, int> parents_by_qubit(int node_id) {
    unordered_map<int, int> relation;
    for (auto& term : nodes[node_id].product.terms) {
      vector<int> parents_involving_q;
      for (auto parent_id : nodes[node_id].parents) {
        if (nodes[parent_id].involves_qubit(term.qubit)) { parents_involving_q.push_back(parent_id); }
      }
      if (!parents_involving_q.empty()) { relation[term.qubit] = youngest_parent(parents_involving_q); }
    }
    return relation;
  }

  unordered_map<int, int> children_by_qubit(int node_id) {
    unordered_map<int, int> relation;
    for (auto& term : nodes[node_id].product.terms) {
      vector<int> children_involving_q;
      for (auto child_id : nodes[node_id].children) {
        if (nodes[child_id].involves_qubit(term.qubit)) { children_involving_q.push_back(child_id); }
      }
      if (!children_involving_q.empty()) { relation[term.qubit] = oldest_child(children_involving_q); }
    }
    return relation;
  }

  // Swap two nodes in the graph
  void swap_nodes(int node_id, int parent_id) {
    /*
          If commuting a clifford through, update the node products beforehand.

          grandparents -> parent -> node -> children

                               |-------?---------v
          grandparents -?-> node -> *parent -?-> children
                     |------?--------^
    */

    if (nodes[node_id].children.contains(parent_id)) {
      swap(node_id, parent_id);
      DBG("    parent is child, swapped: " << nodes[node_id] << " " << nodes[parent_id] << "\n");
    }
    // Find the parents associated with each of node's qubits
    unordered_map<int, int> parent_parents_by_qubit = parents_by_qubit(parent_id);
    unordered_map<int, int> node_children_by_qubit = children_by_qubit(node_id);
    // DBG("    parent_parents_by_qubit " << parent_parents_by_qubit << " node_children_by_qubit " << node_children_by_qubit
    //                                    << "\n");

    nodes[parent_id].children.erase(node_id);
    nodes[parent_id].parents.insert(node_id);
    nodes[node_id].parents.erase(parent_id);
    nodes[node_id].children.insert(parent_id);

    // Only shared qubits need to be updated
    set<int> node_qubits;
    for (auto& term : nodes[node_id].product.terms) {
      node_qubits.insert(term.qubit);
    }
    set<int> shared_qubits;
    for (auto& term : nodes[parent_id].product.terms) {
      if (node_qubits.contains(term.qubit)) { shared_qubits.insert(term.qubit); }
    }
    DBG("    shared_qubits " << shared_qubits << "\n");
    for (auto qubit : shared_qubits) {
      DBG("      check qubit " << qubit << "\n");
      // What grandparents should now point at node?
      {
        auto it = parent_parents_by_qubit.find(qubit);
        if (it != parent_parents_by_qubit.end()) {
          // Update the relationship between grandparent and parent
          int grandparent_id = it->second;
          if (nodes[grandparent_id].children.contains(parent_id)) {
            // Qubits that relate grandparent and parent
            vector<int> related_qubits;
            for (auto [q, n] : children_by_qubit(grandparent_id)) {
              if (n == parent_id) { related_qubits.push_back(q); }
            }
            bool all_in = true;
            for (auto q : related_qubits) {
              if (!nodes[node_id].involves_qubit(q)) {
                all_in = false;
                break;
              }
            }
            if (all_in) {
              nodes[grandparent_id].children.erase(parent_id);
              nodes[parent_id].parents.erase(grandparent_id);
            }
          }
          nodes[grandparent_id].children.insert(node_id);
          nodes[node_id].parents.insert(grandparent_id);
        }
      }
      // What children should now be pointed to by parent?
      {
        auto it = node_children_by_qubit.find(qubit);
        if (it != node_children_by_qubit.end()) {
          // Update the relationship between node and child
          int child_id = it->second;
          if (nodes[node_id].children.contains(child_id)) {
            // Qubits that relate node and child
            vector<int> related_qubits;
            for (auto [q, n] : parents_by_qubit(child_id)) {
              if (n == node_id) { related_qubits.push_back(q); }
            }
            bool all_in = true;
            for (auto q : related_qubits) {
              if (!nodes[parent_id].involves_qubit(q)) {
                all_in = false;
                break;
              }
            }
            if (all_in) {
              nodes[node_id].children.erase(child_id);
              nodes[child_id].parents.erase(node_id);
            }
          }
          nodes[parent_id].children.insert(child_id);
          nodes[child_id].parents.insert(parent_id);
        }
      }
    }
    DBG("swap order " << node_id << " " << parent_id << " " << topological_order[node_id] << " " << topological_order[parent_id]
                      << "\n");
    // Swap order
    swap(topological_order[node_id], topological_order[parent_id]);
    // Update roots
    if (roots.contains(parent_id)) {
      roots.erase(parent_id);
      roots.insert(node_id);
    } else if (nodes[node_id].parents.empty()) {
      roots.insert(node_id);
    }
  }

  // Topological sort starting at node
  void update_topological_order_starting_at(int node_id) {
    update_topo_timer.start();
    int offset = topological_order[node_id];
    int num_nodes = topological_order.size();
    // Step 1: Identify affected nodes
    vector<int> indegrees(num_nodes, -1);
    for (int ni = 0; ni < num_nodes; ni++) {
      if (topological_order[ni] >= offset) { indegrees[ni] = 0; }
    }
    // Step 2: Compute in-degrees inside the affected subgraph
    for (int ni = 0; ni < num_nodes; ni++) {
      if (indegrees[ni] != -1) {
        for (auto child_id : nodes[ni].children) {
          if (indegrees[child_id] != -1) { indegrees[child_id]++; }
        }
      }
    }
    // Step 3: Start sorting from all zero in-degree nodes in affected subgraph
    vector<int> new_order;
    queue<int> q;
    for (int ni = 0; ni < num_nodes; ni++) {
      if (indegrees[ni] == 0) { q.push(ni); }
    }
    while (!q.empty()) {
      assert(q.size() <= num_nodes);
      node_id = q.front();
      q.pop();
      new_order.push_back(node_id);
      for (auto child_id : nodes[node_id].children) {
        if (indegrees[child_id] != -1) {
          indegrees[child_id]--;
          if (indegrees[child_id] == 0) { q.push(child_id); }
        }
      }
    }
    // Step 4: Update the topological order map
    for (int ni = 0; ni < new_order.size(); ni++) {
      topological_order[new_order[ni]] = ni + offset;
    }
    update_topo_timer.stop();
  }

  // Commute a clifford operator to the right past a child node.
  void commute_clifford_right(int clifford_id, int node_id) {
    if (!nodes[clifford_id].is_clifford()) { return; }
    auto& clifford = nodes[clifford_id];
    auto& node = nodes[node_id];
    PauliProduct new_node_prod = clifford.product.commute_right(node.product);
    DBG("  new product " << new_node_prod << " old product " << node.product << "\n");
    node.product = new_node_prod;
    DBG("  before swap " << nodes[clifford_id] << " " << nodes[node_id] << "\n");
    DBG("  topo order before " << topological_order << "\n");
    swap_nodes(clifford_id, node_id);
    DBG("  after swap " << nodes[clifford_id] << " " << nodes[node_id] << "\n");
    DBG("  topo order after " << topological_order << "\n");
    //   If node or clifford has a higher topological order than any of their children, we must recompute the topological order
    bool do_update = false;
    for (auto child_id : node.children) {
      if (topological_order[child_id] > topological_order[node_id]) {
        do_update = true;
        break;
      }
    }
    for (auto child_id : clifford.children) {
      if (topological_order[child_id] > topological_order[clifford_id]) {
        do_update = true;
        break;
      }
    }
    if (do_update) {
      DBG("  Updating topo order starting at " << node_id << "\n");
      update_topological_order_starting_at(node_id);
      DBG("  topo order update " << topological_order << "\n");
    }
  }

  // Commute Clifford operators to the right past non-Clifford operators
  void commute_all_cliffords() {
    Timer timer(__func__);
    set<int> uncommuted_noncliffords;
    for (auto& node : nodes) {
      is_uncommuted_nonclifford(node.id, uncommuted_noncliffords);
    }
    if (uncommuted_noncliffords.empty()) {
      cout << "No uncommuted non-Cliffords\n";
      return;
    }

    cout << "Commuting " << uncommuted_noncliffords.size() << " uncommuted noncliffords\n";
#ifdef DBGTRACE
    for (auto& node : nodes) {
      DBG("node " << node.id << " clifford " << (node.is_clifford() ? "True" : "False") << "\n");
    }
    for (auto nonclifford : uncommuted_noncliffords) {
      DBG("  " << nonclifford << "\n");
    }
    exit(0);
#endif

    int num_commuted = 0;
    int loops = 0;
    int num_uncommuted = uncommuted_noncliffords.size();
    int update_tick = (double)num_uncommuted / 100.0;
    int next_tick = update_tick;

    while (uncommuted_noncliffords.size() > 0) {
      DBG("LOOP " << loops << "\n");
#ifndef DBGTRACE
      if (num_commuted >= next_tick) {
        cout << (int)round(num_commuted * 100.0 / num_uncommuted) << " ";
        cout.flush();
        next_tick = num_commuted + update_tick;
      }
#endif
      set<int> finished_noncliffords;
      for (auto node_id : uncommuted_noncliffords) {
        auto& node = nodes[node_id];
        if (done_commuting_nonclifford(node_id, uncommuted_noncliffords)) {
          finished_noncliffords.insert(node_id);
          DBG("add finished nonclifford " << node_id << "\n");
          continue;
        }
        vector<int> parent_cliffords = get_valid_parent_cliffords(node_id);
        DBG("node_id " << node_id << " valid parents " << parent_cliffords << " parents " << nodes[node_id].parents << "\n");
        if (parent_cliffords.empty()) { continue; }
        // check for loops
        for (auto parent_id : node.parents) {
          if (node.children.contains(parent_id)) { throw runtime_error(to_string(__LINE__) + ": loop detected"); }
        }
        int parent_id = youngest_parent(parent_cliffords);
#ifdef DBGTRACE
        if (parent_cliffords.size() > 1) {
          DBG("youngest parent " << parent_id << "\n");
          for (auto pid : parent_cliffords) {
            DBG("parent id " << pid << " topo " << topological_order[pid] << "\n");
          }
        }
#endif
        DBG("Commute " << nodes[parent_id] << " past " << nodes[node_id] << "\n");
        commute_clifford_right(parent_id, node_id);
      }
      for (auto nonclifford_id : finished_noncliffords) {
        DBG("finished nonclifford " << nonclifford_id << "\n");
        uncommuted_noncliffords.erase(nonclifford_id);
      }
      num_commuted += finished_noncliffords.size();
      loops++;
      DBG("num commuted " << num_commuted << "\n");
#ifdef DBGTRACE
      // if (loops == 51) { break; }
#endif
    }
    cout << "\n";
  }
};

int main(int argc, char* argv[]) {
  signal(SIGABRT, signal_handler);
  signal(SIGSEGV, signal_handler);
  Timer timer(__func__);
  PauliProductDAG dag;
  string fname(argv[1]);
  dag.load_from_file(fname);
  {
    ofstream f(fname + "-loaded.txt");
    f << dag;
    f.close();
  }
  dag.commute_all_cliffords();
  {
    ofstream f(fname + "-transpiled.txt");
    f << dag;
    f.close();
  }
  update_topo_timer.done();
}
