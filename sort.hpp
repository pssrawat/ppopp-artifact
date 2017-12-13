#ifndef __SORT_HPP__
#define __SORT_HPP__
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <cassert>
#include <sstream>
#include <algorithm>

using namespace std;

// Sort the metrics in the order of appearance of indices
// This one does not use second level metric at all
inline bool sort_order_metric_a0 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_a0_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

// Visit in order, avoid the newly added metrics
inline bool sort_order_metric_b0 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b0_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

// Visit in order
inline bool sort_order_metric_b1 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b1_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b2 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b2_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b3 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b3_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b4 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b4_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b5 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_order_metric_b5_ilp (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<16>(a) != get<16>(b)) return (get<16>(a) > get<16>(b));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<17>(a) != get<17>(b)) return (get<17>(a) > get<17>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	if (get<18>(a) != get<18>(b)) return (get<18>(a) > get<18>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

// For jacobi-like problems
inline bool sort_order_metric_b6 (tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> a, tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) > get<5>(b));
	if (get<7>(a) != get<7>(b)) return (get<7>(a) > get<7>(b));
	if (get<6>(a) != get<6>(b)) return (get<6>(a) > get<6>(b));
	if (get<8>(a) != get<8>(b)) return (get<8>(a) > get<8>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<9>(a)+get<10>(a) != get<9>(b)+get<10>(b)) return ((get<9>(a)+get<10>(a)) > (get<9>(b)+get<10>(b)));
	if (get<9>(a) != get<9>(b)) return (get<9>(a) > get<9>(b));
	if (get<10>(a) != get<10>(b)) return (get<10>(a) > get<10>(b));
	if (get<15>(a) != get<15>(b)) return (get<15>(a) < get<15>(b));
	if (get<11>(a)+get<12>(a) != get<11>(b)+get<12>(b)) return ((get<11>(a)+get<12>(a)) > (get<11>(b)+get<12>(b)));
	if (get<11>(a) != get<11>(b)) return (get<11>(a) > get<11>(b));
	if (get<12>(a) != get<12>(b)) return (get<12>(a) > get<12>(b));
	if (get<13>(a)+get<14>(a) != get<13>(b)+get<14>(b)) return ((get<13>(a)+get<14>(a)) > (get<13>(b)+get<14>(b)));
	if (get<13>(a) != get<13>(b)) return (get<13>(a) > get<13>(b));
	if (get<14>(a) != get<14>(b)) return (get<14>(a) > get<14>(b));
	// Now find the lexicographical minimum
	vector<int> a_vec = get<19>(a);
	vector<int> b_vec = get<19>(b);
	if (a_vec.size() != b_vec.size()) 
		return (a_vec.size() < b_vec.size());
	if (a_vec.size() > 0 & b_vec.size() > 0) {
		vector<int>::const_iterator j=b_vec.begin();
		for (vector<int>::const_iterator i=a_vec.begin(); i!=a_vec.end(); i++,j++) {
			if (*i != *j) return (*i < *j);
		}
	}
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

// Sorting the spill metric
inline bool sort_spill_metric_a1 (tuple<string, int, int, int, int, int> a, tuple<string, int, int, int, int, int> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) < get<5>(b));
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_spill_metric_a2 (tuple<string, int, int, int, int, int> a, tuple<string, int, int, int, int, int> b) {
	if (get<1>(a)+get<2>(a) != get<1>(b)+get<2>(b)) return ((get<1>(a)+get<2>(a)) > (get<1>(b)+get<2>(b)));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) < get<5>(b));
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_spill_metric_b1 (tuple<string, int, int, int, int, int> a, tuple<string, int, int, int, int, int> b) {
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) < get<5>(b));
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_spill_metric_b2 (tuple<string, int, int, int, int, int> a, tuple<string, int, int, int, int, int> b) {
	if (get<1>(a)+get<2>(a) != get<1>(b)+get<2>(b)) return ((get<1>(a)+get<2>(a)) > (get<1>(b)+get<2>(b)));
	if (get<1>(a) != get<1>(b)) return (get<1>(a) > get<1>(b));
	if (get<2>(a) != get<2>(b)) return (get<2>(a) > get<2>(b));
	if (get<4>(a) != get<4>(b)) return (get<4>(a) > get<4>(b));
	if (get<3>(a) != get<3>(b)) return (get<3>(a) > get<3>(b));
	if (get<5>(a) != get<5>(b)) return (get<5>(a) < get<5>(b));
	// Otherwise compare strings
	string s1 = get<0>(a);
	string s2 = get<0>(b);
	bool ret = (s1.compare (s2) <= 0) ? true : false;
	return ret;
}

inline bool sort_opt_reg_cost (tuple<int,int,vector<int>> a, tuple<int,int,vector<int>> b) {
	if (get<0>(a)-get<1>(a) < 0 && (get<0>(b)-get<1>(b)) < 0) return (get<0>(a)-get<1>(a)) < (get<0>(b)-get<1>(b));
	if (get<0>(a)-get<1>(a) < 0 && (get<0>(b)-get<1>(b)) >= 0) return (get<0>(a)-get<1>(a)) < (get<0>(b)-get<1>(b));
	if (get<0>(a)-get<1>(a) >= 0 && (get<0>(b)-get<1>(b)) < 0) return (get<0>(a)-get<1>(a)) < (get<0>(b)-get<1>(b));
	if (get<1>(a) != get<1>(b)) return get<1>(a) > get<1>(b);
	if (get<0>(a) != get<0>(b)) return get<0>(a) < get<0>(b);
	//if (get<2>(a).size() != get<2>(b).size()) return get<2>(a).size() > get<2>(b).size();
	return false;
}

#endif
