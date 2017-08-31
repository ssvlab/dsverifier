//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef SET_H
#define SET_H

#include <string>

namespace abstract{
typedef unsigned char set_card_lut_t;

class Set {
protected:
    typedef unsigned long* SetBuffer;

public:
    Set(const long size);
    Set(const Set &set);
    ~Set();
    unsigned long blocks(const long len);
    void initializeEmpty(const long len);
    void initializeSet(const long len);
    void clear();
    void set();
    void copy(const Set &set);
    void add(const long elem);
    void remove(const long elem);
    void intersect(const Set &set1,const Set &set2);
    void merge(const Set &set1,const Set &set2);
    void diff(const Set &set1,const Set &set2);
    void setComplement(const Set &set);
    void addComplement(const Set &set);
    int isSubset(const Set &set);

    /// Indicates if an element is in the set
    /// @param elem element to check for
    /// @return 1 if the element is inside the set 0 if not
    int member(const long elem) const;

    long cardinality() const;
    void logSet(const std::string name, const bool endline=true) const;

protected:
    void initialize(const long len);

public:
    int m_size;
    int m_bufferSize;
    SetBuffer m_pBuffer;
    static bool ms_trace_set;
};
}
#endif//SET_H
