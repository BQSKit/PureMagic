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
#include <stack>
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
  if (traceon) { cerr << x; }
#else
// #define DBG 0 && cout
#define DBG(x)
#endif

static IntermittentTimer update_topo_timer("update_topo");
static IntermittentTimer swap_nodes_timer("swap_nodes");

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

class PauliProductDAG {
private:
  vector<PauliProduct> products;
  vector<unordered_set<int>> children;
  vector<unordered_set<int>> parents;
  unordered_set<int> roots;
  vector<int> topological_order;

  int max_qubit = 0;
  long topo_steps = 0;
  int update_topo_calls = 0;
  int num_nodes = 0;

  set<int> get_sorted(const unordered_set<int>& uset) const {
    set<int> sset;
    for (auto i : uset) {
      sset.insert(i);
    }
    return sset;
  }

  void load_node_from_string(string& s, int node_id) {
    const int NUM_SPACE_TOKENS = 4;
    auto tokens = split_string(s, '\t');
    if (tokens.size() != NUM_SPACE_TOKENS) {
      throw runtime_error(to_string(__LINE__) + ": Incorrect number of tokens: expected " + to_string(NUM_SPACE_TOKENS) +
                          " but got " + to_string(tokens.size()) + " for line:\n" + s);
    }
    try {
      int id = stoi(tokens[0]);
      assert(id == node_id);
    } catch (const exception& ex) {
      cerr << __LINE__ << ": Exception: " << ex.what() << " s " << s << " " << tokens[0] << "\n";
      throw ex;
    }
    products[node_id].load_from_string(tokens[1]);
    children[node_id] = parse_id_list<unordered_set<int>>(tokens[2]);
    parents[node_id] = parse_id_list<unordered_set<int>>(tokens[3]);
  }

  template <typename T>
  T parse_id_list(const string& s) {
    if (s == "[]") { return {}; }
    T ids;
    if (s == "set()") { return ids; }
    auto tokens = split_string(s.substr(1), ',');
    for (auto& token : tokens) {
      try {
        ids.insert(stoi(token));
      } catch (const exception& ex) {
        cerr << __LINE__ << ": Exception: " << ex.what() << " \"" << s << "\" \"" << token << "\"\n";
        throw ex;
      }
    }
    return ids;
  }

  bool is_root(int node_id) { return parents[node_id].empty(); }

  bool is_clifford(int node_id) { return products[node_id].is_clifford; }

  bool involves_qubit(int node_id, int qubit) {
    for (auto& term : products[node_id].terms) {
      if (term.qubit == qubit) { return true; }
    }
    return false;
  }

  bool is_bad_topo_order(int node_id) {
    for (auto child_id : children[node_id]) {
      if (topological_order[child_id] < topological_order[node_id]) { return true; }
    }
    for (auto parent_id : parents[node_id]) {
      if (topological_order[parent_id] > topological_order[node_id]) { return true; }
    }
    return false;
  }

  bool is_uncommuted_nonclifford(int node_id, set<int>& uncommuted_noncliffords) {
    if (is_clifford(node_id)) { return false; }
    if (is_root(node_id)) { return true; }
    if (uncommuted_noncliffords.contains(node_id)) { return true; }
    if (children[node_id].empty()) {
      uncommuted_noncliffords.insert(node_id);
      return true;
    }
    for (auto child_id : children[node_id]) {
      if (is_clifford(child_id) || is_uncommuted_nonclifford(child_id, uncommuted_noncliffords)) {
        uncommuted_noncliffords.insert(node_id);
        return true;
      }
    }
    return false;
  }

  bool done_commuting_nonclifford(int node_id, set<int>& uncommuted_noncliffords) {
    for (auto parent_id : parents[node_id]) {
      if (is_clifford(parent_id) || uncommuted_noncliffords.contains(parent_id)) { return false; }
    }
    return true;
  }

  // Check if an indirect forward path exists from u -> ... -> v
  bool indirect_path_exists(int u, int v) {
    if (u == v) { return true; }
    assert(!is_bad_topo_order(u) && !is_bad_topo_order(v));
    int topo_index_u = topological_order[u];
    int topo_index_v = topological_order[v];
    if (topo_index_u > topo_index_v) { return false; }
    // BFS traversal from u's children that are not v
    queue<int> q;
    unordered_set<int> visited;
    for (auto child_id : children[u]) {
      if (child_id != v) {
        q.push(child_id);
        visited.insert(child_id);
      }
    }
    while (!q.empty()) {
      int node_id = q.front();
      q.pop();
      for (auto child_id : children[node_id]) {
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
    for (auto parent_id : parents[node_id]) {
      if (!is_clifford(parent_id)) { continue; }
      if (!indirect_path_exists(parent_id, node_id)) { parent_cliffords.push_back(parent_id); }
    }
    return parent_cliffords;
  }

  int youngest_node(const vector<int>& related_nodes) {
    return *max_element(related_nodes.begin(), related_nodes.end(),
                        [topo_order = this->topological_order](const int& x, const int& y) {
                          return topo_order[x] < topo_order[y];
                        });
  }

  unordered_map<int, int> relations_by_qubit(int node_id, bool from_children) {
    unordered_map<int, int> relation_map;
    unordered_set<int>& relations = (from_children ? children[node_id] : parents[node_id]);
    for (auto& term : products[node_id].terms) {
      int selected_id = -1;
      int selected_order = 0;
      for (auto relation_id : relations) {
        if (involves_qubit(relation_id, term.qubit)) {
          if ((selected_id == -1) || (from_children && topological_order[relation_id] < selected_order) ||
              (!from_children && topological_order[relation_id] > selected_order)) {
            selected_id = relation_id;
            selected_order = topological_order[relation_id];
          }
        }
      }
      if (selected_id != -1) { relation_map[term.qubit] = selected_id; }
    }
    return relation_map;
  }

  void erase_related(int grandparent_id, int parent_id, int node_id, bool from_children) {
    DBG("Erasing related " << grandparent_id << " " << parent_id << " " << node_id << " from_children "
                           << (from_children ? "true" : "false") << "\n");
    if (!children[grandparent_id].contains(parent_id)) { return; }
    vector<int> related_qubits;
    if (from_children) {
      for (auto [q, n] : relations_by_qubit(grandparent_id, true)) {
        if (n == parent_id) { related_qubits.push_back(q); }
      }
    } else {
      for (auto [q, n] : relations_by_qubit(parent_id, false)) {
        if (n == grandparent_id) { related_qubits.push_back(q); }
      }
    }

    bool all_in = true;
    for (auto q : related_qubits) {
      if (!involves_qubit(node_id, q)) {
        all_in = false;
        break;
      }
    }
    if (all_in) {
      children[grandparent_id].erase(parent_id);
      parents[parent_id].erase(grandparent_id);
    }
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
    swap_nodes_timer.start();
    if (children[node_id].contains(parent_id)) {
      swap(node_id, parent_id);
      DBG("  parent is child, swapped: " << node_id << " " << parent_id << "\n");
    }
    // Find the parents associated with each of node's qubits
    unordered_map<int, int> parent_parents_by_qubit = relations_by_qubit(parent_id, false);
    unordered_map<int, int> node_children_by_qubit = relations_by_qubit(node_id, true);
    // DBG("    parent_parents_by_qubit " << parent_parents_by_qubit << " node_children_by_qubit " << node_children_by_qubit
    //                                    << "\n");

    children[parent_id].erase(node_id);
    parents[parent_id].insert(node_id);
    parents[node_id].erase(parent_id);
    children[node_id].insert(parent_id);

    // Only shared qubits need to be updated
    unordered_set<int> node_qubits;
    for (auto& term : products[node_id].terms) {
      node_qubits.insert(term.qubit);
    }
    vector<int> shared_qubits;
    for (auto& term : products[parent_id].terms) {
      if (node_qubits.contains(term.qubit)) { shared_qubits.push_back(term.qubit); }
    }

    // DBG("    shared_qubits " << shared_qubits << "\n");
    for (auto qubit : shared_qubits) {
      // DBG("      check qubit " << qubit << "\n");
      //  What grandparents should now point at node?
      {
        auto it = parent_parents_by_qubit.find(qubit);
        if (it != parent_parents_by_qubit.end()) {
          // Update the relationship between grandparent and parent
          int grandparent_id = it->second;
          // Qubits that relate grandparent and parent
          erase_related(grandparent_id, parent_id, node_id, true);
          children[grandparent_id].insert(node_id);
          parents[node_id].insert(grandparent_id);
        }
      }
      // What children should now be pointed to by parent?
      {
        auto it = node_children_by_qubit.find(qubit);
        if (it != node_children_by_qubit.end()) {
          // Update the relationship between node and child
          int child_id = it->second;
          // Qubits that relate node and child
          erase_related(node_id, child_id, parent_id, false);
          children[parent_id].insert(child_id);
          parents[child_id].insert(parent_id);
        }
      }
    }
    // DBG("swap order " << node_id << " " << parent_id << " " << topological_order[node_id] << " " <<
    // topological_order[parent_id]
    //                   << "\n");
    //  Swap order
    swap(topological_order[node_id], topological_order[parent_id]);
    // Update roots
    if (roots.contains(parent_id)) {
      roots.erase(parent_id);
      roots.insert(node_id);
    } else if (parents[node_id].empty()) {
      roots.insert(node_id);
    }
    swap_nodes_timer.stop();
  }

  // Topological sort starting at node
  void update_topological_order_starting_at(int node_id) {
    DBG("Updating topological order starting at " << node_id << "\n");
    DBG("Current topo order: " << topological_order << "\n");
    update_topo_calls++;
    update_topo_timer.start();
    int offset = topological_order[node_id];
    int num_nodes = topological_order.size();

    vector<int> indegrees(num_nodes, 0);
    //   Step 1: Compute in-degrees inside the affected subgraph
    int subgraph_nodes = 0;
    for (int ni = 0; ni < num_nodes; ni++) {
      if (topological_order[ni] >= offset) {
        subgraph_nodes++;
        for (auto child_id : children[ni]) {
          if (topological_order[child_id] >= offset) { indegrees[child_id]++; }
        }
      }
    }
    // Step 2: Start sorting from all zero in-degree nodes in affected subgraph
    vector<int> new_order;
    new_order.reserve(subgraph_nodes);
    queue<int> q;
    for (int ni = 0; ni < num_nodes; ni++) {
      if (topological_order[ni] >= offset && indegrees[ni] == 0) { q.push(ni); }
    }
    while (!q.empty()) {
      assert(q.size() <= num_nodes);
      node_id = q.front();
      q.pop();
      DBG("Popped node " << node_id << " with topo order " << topological_order[node_id] << "\n");
      new_order.push_back(node_id);
      DBG("Append to new order " << node_id << "\n");
      set<int, decltype(*this)> sorted_children(*this);
      for (auto child_id : children[node_id]) {
        if (topological_order[child_id] >= offset) { sorted_children.insert(child_id); }
      }
      for (auto child_id : sorted_children) {
        // for (auto child_id : children[node_id]) {
        topo_steps++;
        if (topological_order[child_id] >= offset) {
          indegrees[child_id]--;
          if (indegrees[child_id] == 0) {
            q.push(child_id);
            DBG("From " << node_id << " pushed node " << child_id << "\n");
          }
        }
      }
    }
    //   Step 3: Update the topological order map
    for (int ni = 0; ni < new_order.size(); ni++) {
      topological_order[new_order[ni]] = ni + offset;
    }
    DBG("New topo order: " << topological_order << "\n");
    update_topo_timer.stop();
  }

  // Commute a clifford operator to the right past a child node.
  void commute_clifford_right(int clifford_id, int node_id) {
    if (!is_clifford(clifford_id)) { return; }
    PauliProduct new_node_prod = products[clifford_id].commute_right(products[node_id]);
    DBG("Commuting clifford " << clifford_id << " with nonclifford " << node_id << ":\n    " << products[node_id] << " -> "
                              << new_node_prod << "\n    topo order " << topological_order[clifford_id] << " "
                              << topological_order[node_id] << "\n");
    products[node_id] = new_node_prod;
    swap_nodes(clifford_id, node_id);
    DBG("after swap: " << products[clifford_id] << " " << products[node_id] << "\n    topo order "
                       << topological_order[clifford_id] << " " << topological_order[node_id] << "\n");
    if (is_bad_topo_order(node_id) || is_bad_topo_order(clifford_id)) {
      update_topological_order_starting_at(node_id);
      assert(!is_bad_topo_order(node_id) && !is_bad_topo_order(clifford_id));
    }
  }

public:
  bool operator()(const int n1, const int n2) { return topological_order[n1] < topological_order[n2]; }

  void load_from_file(const string& fname) {
    cout << "Loading circuit from " << fname << "\n";
    ifstream f(fname);
    if (!f) { throw runtime_error("Could not open " + fname + "\n"); }
    vector<string> lines;
    string buf;
    // first line is headers
    getline(f, buf);
    while (getline(f, buf)) {
      lines.push_back(buf);
    }
    cout << "Found " << lines.size() << " lines in " << fname << "\n";
    num_nodes = lines.size();
    children.resize(num_nodes);
    parents.resize(num_nodes);
    products.resize(num_nodes);
    topological_order.resize(num_nodes);
    int num_cliffords = 0;
    int num_edges = 0;
    for (int i = 0; i < lines.size(); i++) {
      int node_id = i;
      load_node_from_string(lines[i], node_id);
      if (is_root(node_id)) { roots.insert(node_id); }
      if (is_clifford(node_id)) { num_cliffords++; }
      topological_order[node_id] = node_id;
      for (auto& term : products[node_id].terms) {
        max_qubit = max(max_qubit, term.qubit);
      }
      num_edges += children[node_id].size();
    }
    for (int ni = 0; ni < num_nodes; ni++) {
      assert(!is_bad_topo_order(ni));
    }
    DBG("Topological order: " << topological_order << "\n");
    cout << "Loaded " << num_nodes << " products from " << fname << " of which " << num_cliffords << " are cliffords and "
         << roots.size() << " are roots, with max qubit " << max_qubit << "\n";
    cout << "Forms a dag with " << num_nodes << " nodes and " << num_edges << " edges\n";
  }

  friend ostream& operator<<(ostream& os, const PauliProductDAG& dag) {
    os << "id\tproduct\tchildren\tparents\n";
    for (int i = 0; i < dag.num_nodes; i++) {
      os << i << "\t" << dag.products[i] << "\t" << dag.get_sorted(dag.children[i]) << "\t" << dag.get_sorted(dag.parents[i])
         << "\n";
    }
    return os;
  }

  // Commute Clifford operators to the right past non-Clifford operators
  void commute_all_cliffords() {
    Timer timer(__func__);
    set<int> uncommuted_noncliffords;
    for (int i = 0; i < num_nodes; i++) {
      is_uncommuted_nonclifford(i, uncommuted_noncliffords);
    }
    if (uncommuted_noncliffords.empty()) {
      cout << "No uncommuted non-Cliffords\n";
      return;
    }
    cout << "Commuting " << uncommuted_noncliffords.size() << " uncommuted noncliffords\n";
    int num_commuted = 0;
    int loops = 0;
    int num_uncommuted = uncommuted_noncliffords.size();
    int update_tick = (double)num_uncommuted / 20.0;
    int next_tick = update_tick;

    while (uncommuted_noncliffords.size() > 0) {
      if (num_commuted >= next_tick) {
        cout << (int)round(num_commuted * 100.0 / num_uncommuted) << " ";
        cout.flush();
        next_tick = num_commuted + update_tick;
      }
      vector<int> finished_noncliffords;
      for (auto node_id : uncommuted_noncliffords) {
        if (done_commuting_nonclifford(node_id, uncommuted_noncliffords)) {
          finished_noncliffords.push_back(node_id);
          DBG("Finished commuting nonclifford " << node_id << "\n");
          continue;
        }
        vector<int> parent_cliffords = get_valid_parent_cliffords(node_id);
        // DBG("node_id " << node_id << " valid parents " << parent_cliffords << " parents " << parents[node_id] << "\n");
        if (parent_cliffords.empty()) { continue; }
        // check for loops
        for (auto parent_id : parents[node_id]) {
          if (children[node_id].contains(parent_id)) { throw runtime_error(to_string(__LINE__) + ": loop detected"); }
        }
        int parent_id = youngest_node(parent_cliffords);
        if (parent_cliffords.size() > 1) { DBG("youngest parent " << parent_id << "\n"); }
        commute_clifford_right(parent_id, node_id);
      }
      for (auto nonclifford_id : finished_noncliffords) {
        DBG("Removed nonclifford " << nonclifford_id << "\n");
        uncommuted_noncliffords.erase(nonclifford_id);
      }
      num_commuted += finished_noncliffords.size();
      loops++;
      DBG("Iteration " << loops << ": Commuted " << finished_noncliffords.size() << " noncliffords, remaining "
                       << uncommuted_noncliffords.size() << "\n");
    }
    cout << "\n";
    // now check - if we are a nonclifford, we should have no clifford children
    for (int node_id = 0; node_id < num_nodes; node_id++) {
      if (is_clifford(node_id)) {
        for (auto child_id : children[node_id]) {
          if (!is_clifford(child_id)) {
            cerr << "WARNING: Found clifford " << node_id << " with nonclifford child " << child_id << "\n";
          }
        }
      }
    }
    cout << "There were " << topo_steps << " steps in " << update_topo_calls << " calls to update the topological order\n";
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
    cout << "Saving loaded circuit to " << fname << "-loaded.txt\n";
    ofstream f(fname + "-loaded.txt");
    f << dag;
    f.close();
  }
  dag.commute_all_cliffords();
  {
    cout << "Saving transpiled circuit to " << fname << "-transpiled.txt\n";
    ofstream f(fname + "-transpiled.txt");
    f << dag;
    f.close();
  }
  update_topo_timer.done();
  swap_nodes_timer.done();
}
