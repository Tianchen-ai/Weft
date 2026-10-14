#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

void encoded_read_order(const uint8_t *, int32_t *, int32_t *,
                        const int32_t *, size_t);

int main(void) {
  int32_t a = 3, b = 0, replacement = 7;
  encoded_read_order((const uint8_t *)&a, &a, &b, &replacement, 1);
  printf("a=%d b=%d expected_a=7 expected_b=3\n", a, b);
  return a == 7 && b == 3 ? 0 : 1;
}
