// Copyright 2021 Barysheva Maria
#ifndef MODULES_TASK_3_BARYSHEVA_M_RADIX_BATCHER_RADIX_BATCHER_H_
#define MODULES_TASK_3_BARYSHEVA_M_RADIX_BATCHER_RADIX_BATCHER_H_

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

double *getRandomVector(const int size);
void RadixSortParallel(double **arr, int size);

template <class T>
void createCounters(T *data, uint64_t *counters, uint64_t N) {
  memset(counters, 0, 256 * sizeof(T) * sizeof(uint64_t));

  uint8_t *bp = reinterpret_cast<uint8_t *>(data);
  uint8_t *dataEnd = reinterpret_cast<uint8_t *>(data + N);
  uint16_t i;

  while (bp != dataEnd) {
    for (i = 0; i < sizeof(T); i++) counters[256 * i + *bp++]++;
  }
}

template <class T>
void radixPass(uint16_t Offset, uint64_t N, T *source, T *dest,
               uint64_t *count) {
  T *sp;
  uint64_t s, c, i, *cp;
  uint8_t *bp;

  s = 0;
  cp = count;
  for (i = 256; i > 0; --i, ++cp) {
    c = *cp;
    *cp = s;
    s += c;
  }

  bp = reinterpret_cast<uint8_t *>(source) + Offset;
  sp = source;
  for (i = N; i > 0; --i, bp += sizeof(T), ++sp) {
    cp = count + *bp;
    dest[*cp] = *sp;
    ++(*cp);
  }
}

template <class T>
void floatRadixLastPass(uint16_t Offset, uint64_t N, T *source, T *dest,
                        uint64_t *count) {
  T *sp;
  uint64_t s, c, i, *cp;
  uint8_t *bp;

  uint64_t numNeg = 0;
  for (i = 128; i < 256; i++) numNeg += count[i];

  s = numNeg;
  cp = count;
  for (i = 0; i < 128; ++i, ++cp) {
    c = *cp;
    *cp = s;
    s += c;
  }

  s = count[255] = 0;
  cp = count + 254;
  for (i = 254; i >= 128; --i, --cp) {
    *cp += s;
    s = *cp;
  }

  bp = reinterpret_cast<uint8_t *>(source) + Offset;
  sp = source;
  for (i = N; i > 0; --i, bp += sizeof(T), ++sp) {
    cp = count + *bp;
    if (*bp < 128)
      dest[(*cp)++] = *sp;
    else
      dest[--(*cp)] = *sp;
  }
}

template <class T>
void floatRadixSort(T **in, uint64_t N) {
  T *out = new T[N];
  uint16_t i;

  uint64_t *counters = new uint64_t[sizeof(T) * 256], *count;
  createCounters((*in), counters, N);

  for (i = 0; i < sizeof(T) - 1; i++) {
    count = counters + 256 * i;
    if (count[0] == N) continue;
    radixPass(i, N, (*in), out, count);
    std::swap((*in), out);
  }
  count = counters + 256 * i;
  floatRadixLastPass(i, N, (*in), out, count);

  delete (*in);
  (*in) = out;
  delete[] counters;
}

#endif  // MODULES_TASK_3_BARYSHEVA_M_RADIX_BATCHER_RADIX_BATCHER_H_
