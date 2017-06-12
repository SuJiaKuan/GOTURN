// Visualize the tracker from video stream

#include <iostream>
#include <cstdlib>

#include "network/regressor.h"
#include "tracker/tracker.h"
#include "tracker/tracker_manager.h"

using std::string;

const bool show_intermediate_output = false;

int main (int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " [gpu_id] [pauseval]" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  const string& model_file   = argv[1];
  const string& trained_file = argv[2];

  int gpu_id = 0;
  if (argc >= 4) {
    gpu_id = atoi(argv[3]);
  }

  int pause_val = 1;
  if (argc >= 7) {
    pause_val = atoi(argv[4]);
  }

  std::cout << model_file << " " << trained_file << " " << gpu_id << std::endl;

  // Set up the neural network.
  const bool do_train = false;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);

  Tracker tracker(show_intermediate_output);

  TrackerStreamer tracker_streamer(&regressor, &tracker);
  tracker_streamer.Track(0, 0, 100, 100, pause_val);

  return 0;
}
