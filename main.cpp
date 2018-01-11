#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include "grammar.hpp"
#include "codegen.hpp"

using namespace std;

start_node * grammar::start = NULL;

int main (int argc, char **argv) {
	string outfile ("--out-file");
	string out_name ("out.cu");
	string datatype ("--datatype");
	string data_type ("double");
	string unroll ("--unroll");
    map<string, int> unroll_decls;
	string dist_rhs ("--distribute-rhs");
	bool distribute_rhs = true;
	string heuristic_used ("--heuristic");
	int heuristic = 0;
	string split_accs ("--split");
	bool split = false;
	string top_sort ("--topo-sort");
	bool topo_sort = false;

	if (DEBUG) printf ("filename : %s\n", argv[1]);
	for (int i=2; i<argc; i++) {
		if (outfile.compare (argv[i]) == 0) {
			if (i == argc-1) 
				printf ("Missing output file name, using default out.cu\n");
			else 
				out_name = argv[++i];
		}
		if (datatype.compare (argv[i]) == 0) {
			if (i == argc-1) 
				printf ("Missing datatype, using default double\n");
			else 
				data_type = argv[++i];
		}
		if (dist_rhs.compare (argv[i]) == 0) {
			if (i == argc-1) 
				printf ("Missing distribute rhs value, using default true\n");
			else {
				string tmp = argv[++i];
				distribute_rhs = (tmp.compare ("true") == 0) ? true : false;
			} 
		}
		if (split_accs.compare (argv[i]) == 0) {
			if (i == argc-1) 
				printf ("Missing split acc value, using default false\n");
			else {
				string tmp = argv[++i];
				split = (tmp.compare ("true") == 0) ? true : false;
			} 
		}
		if (top_sort.compare (argv[i]) == 0) {
			if (i == argc-1) 
				printf ("Missing topological sort value, using default false\n");
			else {
				string tmp = argv[++i];
				topo_sort = (tmp.compare ("true") == 0) ? true : false;
			}
		}
		if (heuristic_used.compare (argv[i]) == 0) {
			if (i == argc-1) 
				printf ("Missing heuristic value, using default 0\n");
			else 
				heuristic = atoi (argv[++i]);
		}
		if (unroll.compare (argv[i]) == 0) {
			if (i != argc-1) {
				string tmp = argv[++i];
				while (tmp.find (",") != string::npos) {
					size_t pos = tmp.find (",");
					string uf = tmp.substr(0,pos);
					size_t vpos = uf.find ("=");
					string dimension = 	uf.substr(0,vpos);
					int val = atoi(uf.substr(vpos+1).c_str());
					unroll_decls[dimension] = val;
					tmp = tmp.substr(pos+1);		
				}
                                size_t vpos = tmp.find ("=");
                                string dimension = tmp.substr(0,vpos);
                                int val = atoi(tmp.substr(vpos+1).c_str());
                                unroll_decls[dimension] = val;
			}
		}
	}
	if (DEBUG) printf ("output file : %s\n", out_name.c_str());
	FILE *in = fopen (argv[1], "r");
	string orig_name = "orig_" + out_name;
	ofstream original_out (orig_name.c_str(), ofstream::out);
	ofstream reorder_out (out_name.c_str(), ofstream::out);

	grammar::set_input (in);
	grammar::parse ();
	codegen *sp_gen = new codegen (grammar::start);
	stringstream reorder, original;
	DATA_TYPE gdata_type = DOUBLE; 
	if (data_type.compare ("float") == 0) 
		gdata_type = FLOAT;
	sp_gen->generate_code (reorder, original, unroll_decls, gdata_type, heuristic, distribute_rhs, split, topo_sort);
	original_out << original.rdbuf ();
	reorder_out << reorder.rdbuf ();
	original_out.close ();
	reorder_out.close ();
	fclose (in);
	return 0;
}
