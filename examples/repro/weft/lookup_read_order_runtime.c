#include <stdint.h>
#include <stdio.h>

void lookup_read_order(const int32_t *, int32_t *, int32_t *, int32_t *);

int main(void) {
  int32_t a[4] = {3, 0, 0, 0}, before = 0, after = 0;
  lookup_read_order(a, a, &before, &after);
  printf("before=%d after=%d expected_before=3 expected_after=7\n", before, after);
  return before == 3 && after == 7 ? 0 : 1;
}
