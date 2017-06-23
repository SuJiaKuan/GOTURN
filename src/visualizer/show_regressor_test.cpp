#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "helper/helper.h"
#include "network/regressor.h"

int main (int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " dataset_folder"
              << " [show_result_only]" << std::endl;
    return 1;
  }

  const std::string& root_folder = argv[1];

  int showResultOnly = 0;
  if (argc >= 3) {
    showResultOnly = atoi(argv[2]);
  }

  ::google::InitGoogleLogging(argv[0]);

  // Find all folders for test images.
  std::vector<std::string> folders;
  std::vector<std::string> class_folders;
  find_subfolders(root_folder, &class_folders);

  // Loop over all classification folders
  for (size_t i = 0; i < class_folders.size(); i++) {
    std::vector<std::string> sub_folders;
    std::string class_folder_path = root_folder + "/" + class_folders[i];
    find_subfolders(class_folder_path, &sub_folders);

    // Loop over all sub folders in the classification folder
    for (size_t j = 0; j < sub_folders.size(); j++) {
      std::string folder_path = class_folder_path + "/" + sub_folders[j];
      folders.push_back(folder_path);
    }
  }

  if (!showResultOnly) {
    // Initialize the regressor.
    const std::string model_file = "nets/tracker.prototxt";
    const std::string trained_file = "nets/models/pretrained_model/tracker.caffemodel";
    const int gpu_id = 0;
    const bool do_train = false;
    Regressor regressor(model_file, trained_file, gpu_id, do_train);

    // Loop over all test folders.
    for (size_t index = 0; index < folders.size(); index++) {
      // Get the folder name.
      const std::string& folder_path = folders[index];

      // Read images.
      // search_origin is the entire image for searching object.
      // search_cropped is the cropped image for search_origin.
      // target_cropped is the cropped image for target image.
      cv::Mat search_origin = cv::imread(folder_path + "/search_origin.jpg");
      cv::Mat search_cropped = cv::imread(folder_path + "/search_cropped.jpg");
      cv::Mat target_cropped = cv::imread(folder_path + "/target_cropped.jpg");

      // Estimate the bounding box location of the target, centered and scaled relative to the cropped image.
      BoundingBox bbox;
      regressor.Regress(search_origin, search_cropped, target_cropped, &bbox);

      // Unscale the estimation to the real image size.
      BoundingBox bbox_unscaled;
      bbox.Unscale(search_cropped, &bbox_unscaled);

      // Draw red bouding box on the estimated location.
      bbox_unscaled.Draw(255, 0, 0, &search_cropped);

      // Show the estimated location
      std::cout << folder_path
          << " " << bbox_unscaled.x1_ << " " << bbox_unscaled.y1_
          << " " << bbox_unscaled.x2_ << " " << bbox_unscaled.y2_
          << std::endl;

      // Write the result image
      cv::imwrite(folder_path + "/result.jpg", search_cropped);
    }
  }

  // Show all result images.
  for (size_t index = 0; index < folders.size(); index++) {
    // Get the folder name.
    const std::string& folder_path = folders[index];

    // Read images.
    // target_cropped is the cropped image for target image.
    // result is the cropped search image with estimated location of target.
    cv::Mat target_cropped = cv::imread(folder_path + "/target_cropped.jpg");
    cv::Mat result = cv::imread(folder_path + "/result.jpg");

    // Setup the window size.
    const int padding_top = 50;
    const int dstWidth = target_cropped.cols * 2;
    const int dstHeight = target_cropped.rows + padding_top;

    // Put images together
    cv::Mat dst = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat targetROI = dst(cv::Rect(0, padding_top, target_cropped.cols, target_cropped.rows));
    target_cropped.copyTo(targetROI);
    targetROI = dst(cv::Rect(target_cropped.cols, padding_top, target_cropped.cols, target_cropped.rows));
    result.copyTo(targetROI);

    // Write down the title.
    cv::putText(dst, folder_path, cv::Point(target_cropped.cols / 2, padding_top / 2), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 255, 0), 2.0);

    // Show the result.
    cv::namedWindow("Regressor Test", cv::WINDOW_AUTOSIZE);
    cv::imshow("Regressor Test", dst);
    cv::waitKey(0);
  }

  return 0;
}
