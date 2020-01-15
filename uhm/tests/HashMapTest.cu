#include "../UHM.cuh"
#include <iostream>

using namespace uhm;
using namespace std;

int main() {
  cerr << "Starting Test\n";
  HashMap<int, int> h(2);
  cerr << "Created hash map\n";
  cerr << "Geting 1\n";
  uhm::optional<int> ret = h.get(1);
  if (ret.isNullopt()) {
    cerr << "Got what was expected\n";
  } else {
    cerr << "ERR got " << *ret << "\n";
    return 1;
  }
  cerr << "Inserting 1\n";
  bool b = h.insert(1, 1);
  cerr << "Retval " << b << endl;
  if (b == false) {
    return 1;
  }
  cerr << "Inserted 1\n";
  cerr << "Geting 1\n";
  ret = h.get(1);
  if (ret.isNullopt()) {
    cerr << "Did not get what was expected\n";
  } else {
    cerr << "GET Returned " << *ret << endl;
  }
  cerr << "Got 1\n";

  return 0;
}