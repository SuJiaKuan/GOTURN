// Visualize the tracker from video stream

#include "network/regressor.h"
#include "tracker/tracker.h"
#include "tracker/tracker_manager.h"

using std::string;

const bool show_intermediate_output = false;

int main (int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel stream_device"
              << " [gpu_id] [pauseval]" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  const string& model_file   = argv[1];
  const string& trained_file = argv[2];
  const string& stream_device = argv[3];

  int gpu_id = 0;
  if (argc >= 5) {
    gpu_id = atoi(argv[4]);
  }

  int pause_val = 1;
  if (argc >= 6) {
    pause_val = atoi(argv[5]);
  }

  // Set up the neural network.
  const bool do_train = false;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);

  // Set up the tracker
  Tracker tracker(show_intermediate_output);

  // Start tracking
  TrackerStreamer tracker_streamer(&regressor, &tracker);
  tracker_streamer.Track(stream_device, pause_val);

  return 0;
}
