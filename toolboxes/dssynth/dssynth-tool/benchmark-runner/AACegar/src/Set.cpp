//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include <limits.h>
#include <stdio.h>
#include <fstream>

#include "Set.h"

namespace abstract{
#define SETBITS (sizeof(long) * CHAR_BIT)
/* (Number of chars in a long) * (number of bits in a char) */

const long setBitShift=(SETBITS==64) ? 6 : ((SETBITS==32) ? 5 : 4);
const long setBitMask=SETBITS-1;

/* Caution!!!
   Bremner's technique depends on the assumption that CHAR_BIT == 8.
*/

#define LUTBLOCKS(set) (((m_size/*>>setBitShift*/)/SETBITS+1)*(sizeof(long)/sizeof(set_card_lut_t)))

static unsigned char set_card_lut[]={
0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};
/* End of Definitions for optimized set_card */


bool Set::ms_trace_set=false;

Set::Set(const long size) : m_size(0),m_bufferSize(0), m_pBuffer(NULL)
{
  initializeEmpty(size);
}

Set::Set(const Set &set) : m_size(0),m_bufferSize(0), m_pBuffer(NULL)
{
  copy(set);
}

inline unsigned long Set::blocks(const long len)
{
  if (len>0) return ((len-1)/SETBITS)+2;
  return 1;
}

inline void Set::initialize(const long size)
/* Make a set with a given bit lengths  */
{
  if (m_size<size) {
    long bufferSize=blocks(size);
    if (bufferSize>m_bufferSize) {
      unsigned long* pBuffer=new unsigned long[bufferSize];
      if (m_pBuffer) {
        for (int i=0;i<m_bufferSize;i++) pBuffer[i]=m_pBuffer[i];
        delete m_pBuffer;
      }
      m_pBuffer=pBuffer;
      m_bufferSize=bufferSize;
    }
    m_size=size;
  }
}

void Set::initializeEmpty(long size)
/* Make a set with a given bit lengths  */
{
  initialize(size);
  clear();
}

void Set::initializeSet(long size)
/* Make a set with a given bit lengths  */
{
  initialize(size);
  set();
}

Set::~Set()
{
  delete m_pBuffer;
}

void Set::clear()
{
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]=0;
}

void Set::set()
{
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]=-1;
}


void Set::copy(const Set &set)
{
  initialize(set.m_size);
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]=set.m_pBuffer[i];
}

void Set::add(long elem)
{
  if (elem>=m_size) initialize(elem+1);
  m_pBuffer[(elem>>setBitShift)]|=(1 << (elem&setBitMask));
}

void Set::remove(long elem)
{
  if (elem<m_size) m_pBuffer[(elem>>setBitShift)]&=~(1 << (elem&setBitMask));
}

void Set::intersect(const Set &set1,const Set &set2)
/* Set intersection, assuming set1 and set2 have the same length as set */
{
  SetBuffer pBuffer1=set1.m_pBuffer;
  SetBuffer pBuffer2=set2.m_pBuffer;
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]=(pBuffer1[i] & pBuffer2[i]);
}

void Set::merge(const Set &set1,const Set &set2)
/* Set union,assuming set1 and set2 have the same length as set */
{
  SetBuffer pBuffer1=set1.m_pBuffer;
  SetBuffer pBuffer2=set2.m_pBuffer;
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]=pBuffer1[i] | pBuffer2[i];
}

void Set::diff(const Set &set1,const Set &set2)
/* Set difference se1/set2, assuming set1 and set2 have the same length as set */
{
  SetBuffer pBuffer1=set1.m_pBuffer;
  SetBuffer pBuffer2=set2.m_pBuffer;
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]=pBuffer1[i] & (~pBuffer2[i]);
}

void Set::setComplement(const Set &set)
/* set[] will be set to the complement of set1[] */
{
  SetBuffer pBuffer=set.m_pBuffer;
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]= ~pBuffer[i];
}

void Set::addComplement(const Set &set)
/* set[] will be set to the complement of set1[] */
{
  SetBuffer pBuffer=set.m_pBuffer;
  for (int i=0;i<m_bufferSize;i++) m_pBuffer[i]|= ~pBuffer[i];
}

int Set::isSubset(const Set &set)
/* Set containment check, this <= pSet */
{
  SetBuffer pBuffer=set.m_pBuffer;
  for (int i=0;i<m_bufferSize;i++) {
    if ((m_pBuffer[i] | pBuffer[i])!=pBuffer[i]) return 0;
  }
  return 1;
}

/// Indicates if an element is in the set
int Set::member(const long elem) const
{
  if (elem<m_size) {
    if (m_pBuffer[elem>>setBitShift]&(1<<(elem&setBitMask))) return 1;
  }
  return 0;
}

long Set::cardinality() const
/* set cardinality, modified by David Bremner bremner@cs.mcgill.ca
   to optimize for speed. */
{
  long car=0;
  set_card_lut_t* p=(set_card_lut_t *)m_pBuffer;
  for (unsigned long block=0; block< LUTBLOCKS(set);block++) {
    car+=set_card_lut[p[block]];
  }
  return car;
}


void Set::logSet(const std::string name,const bool endline) const
{
  if (ms_trace_set) {
    std::ofstream trace;
    trace.open("trace.txt",std::ios_base::app);
    if (trace.is_open()) {
      if (!name.empty()) {
        trace.write(name.data(),name.size());
        trace << " ";
      }
      for (int i=0;i<m_size;i++) trace << member(i);
      if (endline) trace << std::endl;
      trace.close();
    }
  }
}

}
