#include <iostream>
#include <string>
#include <time.h>
#include <boost/filesystem.hpp>
#include "virtual_scanner/virtual_scanner.h"

using namespace std;

void get_all_filenames(vector<string>& _all_filenames, string _filename);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: VirtualScanner.exe <file name/folder name> "
            "[view_num] [flags] [normalize]" << endl;
    return 0;
  }
  string filename(argv[1]);

  int view_num = 6; // scanning view number
  if (argc >= 3) view_num = atoi(argv[2]);

  bool flag = false; // output normal flipping flag
  if (argc >= 4) flag = atoi(argv[3]);

  bool normalize = false; // normalize input meshes
  if (argc >= 5) normalize = atoi(argv[4]);

  vector<string> all_files;
  get_all_filenames(all_files, filename);

  #pragma omp parallel for
  for (int i = 0; i < all_files.size(); i++) {
    clock_t t1 = clock();
    VirtualScanner scanner;
    scanner.scanning(all_files[i], view_num, flag, normalize);
    string out_path  = all_files[i].substr(0, all_files[i].rfind('.'));
    scanner.save_ply(out_path+ ".ply");
    clock_t t2 = clock();

    string messg = all_files[i].substr(all_files[i].rfind('\\') + 1) +
        " done! Time: " + to_string(t2 - t1) + "\n";
    #pragma omp critical
    cout << messg;
  }

  return 0;
}

bool is_convertable_file(string extension) {
  return extension.compare(".off") == 0 || extension.compare(".obj") == 0;
}

void get_all_filenames(vector<string>& _all_filenames, string _filename) {
  using namespace boost::filesystem;

  if (is_regular_file(_filename)) {
    _all_filenames.push_back(_filename);
  } else if (is_directory(_filename)) {
    for (auto& file : recursive_directory_iterator(_filename)) {
      auto extension = file.path().extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
      if (is_regular_file(file) && is_convertable_file(extension)) {
        _all_filenames.push_back(file.path().string());
      }
    }
  }
}
