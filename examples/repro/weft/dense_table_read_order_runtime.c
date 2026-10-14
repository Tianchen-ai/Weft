#include <stdint.h>
#include <stdio.h>

void dense_table_read_order(uint32_t *A, const uint32_t *S, uint32_t *Y);

int main(void) {
  uint32_t table[5] = {3, 11, 13, 17, 19};
  const uint32_t expected[5] = {3, 11, 13, 17, 19};
  const uint32_t replacement[1] = {7};
  uint32_t result[5] = {0};
  dense_table_read_order(table, replacement, result);
  for (unsigned i = 0; i < 5; ++i)
    if (result[i] != expected[i]) {
      fprintf(stderr, "index=%u expected=%u actual=%u\n", i, expected[i], result[i]);
      return 1;
    }
  printf("table_after=%u snapshot_first=%u numeric=exact\n", table[0], result[0]);
  return table[0] == 7 ? 0 : 1;
}
