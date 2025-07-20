#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>

using namespace std;

void signal_handler (int signal) {
    void* array[10];
    size_t size;

    // Get void*'s for all entries currently on the stack
    size = backtrace(array, 100);

    // Print out all the frames to stderr
    std::cerr << "Error: signal " << signal << "\n";
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

static vector<string> split_string (const string& s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

class PauliTerm {
  private:
    char basis;
    int phase;
    int qubit;

  public:
    void load_from_string (const string& s) {
        try {
            if (!s.starts_with("Pauli")) { throw runtime_error("Error loading Pauli product term from string " + s); }
            basis = s[5];
            phase = stoi(s.substr(7, 2));
            qubit = stoi(s.substr(s.length() - 1, 1));
            // cout << s << " basis " << basis << " phase " << phase << " qubit " << qubit << "\n";
        } catch (const exception& ex) {
            cerr << __LINE__ << "Exception: " << ex.what() << " s " << s << " " << s.substr(7, 2) << "\n";
            throw ex;
        }
    }

    friend ostream& operator<<(ostream& os, const PauliTerm& term) {
        os << term.basis << "(" << term.phase << ")" << term.qubit;
        return os;
    }
};

class PauliProduct {
  private:
    vector<PauliTerm> terms;
    int angle_numerator = 1;
    int angle_denominator = 1;

  public:
    void load_from_string (const string& s) {
        auto tokens = split_string(s, '<');
        if (tokens.size() != 2 || !tokens[1].starts_with("Angle")) {
            throw runtime_error(to_string(__LINE__) + ": Error loading Pauli product angle from string " + s);
        }
        auto angle_string = tokens[1].substr(string("Angle(").length());
        auto factors = split_string(angle_string, '/');
        if (factors[0].length() > 2) { angle_numerator = stoi(factors[0].substr(0, factors[0].length() - 2)); }
        if (factors.size() == 2) { angle_denominator = stoi(factors[1].substr(0, factors[1].length() - 2)); }
        auto term_tokens = split_string(tokens[0], '.');
        for (auto& token : term_tokens) {
            PauliTerm term;
            term.load_from_string(token);
            terms.push_back(term);
        }
    }

    friend ostream& operator<<(ostream& os, const PauliProduct& pp) {
        for (auto& term : pp.terms) {
            os << term << ".";
        }
        os << "<" << pp.angle_numerator << "pi/" << pp.angle_denominator << ">";
        return os;
    }
};

class PauliProductDAGNode {
  private:
    int id;
    PauliProduct product;
    vector<int> children;
    vector<int> parents;

    vector<int> parse_id_list (const string& s) {
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
    void load_from_string (string& s) {
        const int NUM_SPACE_TOKENS = 5;
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
        children = parse_id_list(tokens[3]);
        parents = parse_id_list(tokens[4]);
    }

    friend ostream& operator<<(ostream& os, const PauliProductDAGNode& node) {
        os << node.id << " " << node.product << " ";
        for (int i = 0; i < node.children.size(); i++) {
            os << node.children[i];
            if (i < node.children.size() - 1) { os << ","; }
        }
        os << " ";
        for (int i = 0; i < node.parents.size(); i++) {
            os << node.parents[i];
            if (i < node.parents.size() - 1) { os << ","; }
        }
        return os;
    }
};

class PauliProductDAG {
  private:
    vector<PauliProductDAGNode> nodes;

  public:
    void load_from_file (const string& fname) {
        cout << "Loading circuit from " << fname << "\n";
        ifstream f(fname);
        if (!f) { throw runtime_error("Could not open " + fname + "\n"); }
        string buf;
        while (getline(f, buf)) {
            PauliProductDAGNode node;
            node.load_from_string(buf);
            nodes.push_back(node);
        }
        cout << "Loaded " << nodes.size() << " products from " << fname << "\n";
    }

    friend ostream& operator<<(ostream& os, const PauliProductDAG& dag) {
        for (auto& node : dag.nodes) {
            os << node << "\n";
        }
        return os;
    }
};

int main (int argc, char* argv[]) {
    signal(SIGABRT, signal_handler);
    PauliProductDAG dag;
    string fname(argv[1]);
    dag.load_from_file(argv[1]);
    ofstream f("dag-loaded.txt");
    f << dag;
    f.close();
}
