int main(void) {
  __CPROVER_fixedbv[16][8] tmp = 99999999999999999.9999999999999;
  __CPROVER_fixedbv[16][8] tmp2 = tmp;
  __CPROVER_assert(0 == 1, "");

  return 0;
}