#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "helper/helper.h"
#include "helper/image_proc.h"
#include "network/regressor.h"

// Read groud truth from a file
void ReadGroundTruth(const std::string& file_path, BoundingBox* bbox) {
  FILE* gt_file = fopen(file_path.c_str(), "r");
  int x1, y1, x2, y2;

  // Read first line from groud truth file
  fscanf(gt_file, "%d,%d,%d,%d\n", &x1, &y1, &x2, &y2);
  // Close the file
  fclose(gt_file);

  // Read the data into a boudning box
  bbox->x1_ = std::min(x1, x2);
  bbox->y1_ = std::min(y1, y2);
  bbox->x2_ = std::max(x1, x2);
  bbox->y2_ = std::max(y1, y2);
}

// Crop image by detecting the location of bouding box relative to the image
// to make the black padding less.
void CropLessPadImage(
    const BoundingBox& bbox,
    const cv::Mat& image,
    const int& output_width,
    const int& output_height,
    cv::Mat* output_image) {
  // Assign a new image with black background to the output.
  *output_image = cv::Mat(output_height, output_width, image.type(), cv::Scalar(0, 0, 0));

  // Test what quadrant of the center of bouding box locates at relative to the image
  // and decide what ROI should be cropped.
  const int bbox_center_x = (int)bbox.get_center_x();
  const int bbox_center_y = (int)bbox.get_center_y();
  const int img_center_x = (int)(image.cols / 2);
  const int img_center_y = (int)(image.rows / 2);
  int roi_x, roi_y, roi_width, roi_height;
  if (bbox_center_x < img_center_x && bbox_center_y < img_center_y) {
    // 2nd quadrant
    roi_x = (int)bbox.x1_;
    roi_y = (int)bbox.y1_;
  } else if (bbox_center_x < img_center_x && bbox_center_y >= img_center_y) {
    // 3rd quadrant
    roi_x = (int)bbox.x1_;
    roi_y = std::max((int)bbox.y2_ - output_height, 0);
  } else if (bbox_center_x >= img_center_x && bbox_center_y < img_center_y) {
    // 1st quadrant
    roi_x = std::max((int)bbox.x2_ - output_width, 0);
    roi_y = (int)bbox.y1_;
  } else {
    // 4th quadrant
    roi_x = std::max((int)bbox.x2_ - output_width, 0);
    roi_y = std::max((int)bbox.y2_ - output_height, 0);
  }
  roi_width = std::min(output_width, image.cols - roi_x);
  roi_height = std::min(output_height, image.rows - roi_y);

  // Crop original image by ROI and copy to the output iamge.
  cv::Mat image_roi = image(cv::Rect(roi_x, roi_y, roi_width, roi_height));
  cv::Mat output_image_roi = (*output_image)(cv::Rect(0, 0, roi_width, roi_height));
  image_roi.copyTo(output_image_roi);
}

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

      // target_origin is the entire image for tracked object.
      // search_origin is the entire image for searching tracked object.
      // target_bbox is the bouding box of tracked object in target_origin.
      // search_bbox is the bouding box of tracked object in search_origin.
      // target_cropped is the cropped image for target origin.
      // search_cropped is the cropped image for search_origin.
      cv::Mat target_origin = cv::imread(folder_path + "/target.jpg");
      cv::Mat search_origin = cv::imread(folder_path + "/search.jpg");
      BoundingBox target_bbox;
      BoundingBox search_bbox;
      cv::Mat target_cropped;
      cv::Mat search_cropped;

      // Read ground truth from files and write them to bouding boxes.
      ReadGroundTruth(folder_path + "/target_gt.txt", &target_bbox);
      ReadGroundTruth(folder_path + "/search_gt.txt", &search_bbox);

      // Crop target and search images.
      CropPadImage(target_bbox, target_origin, &target_cropped);
      CropLessPadImage(search_bbox, search_origin, target_cropped.cols, target_cropped.rows, &search_cropped);

      // Estimate the bounding box location of the tracked object, centered and scaled relative to the cropped image.
      BoundingBox bbox;
      regressor.Regress(search_origin, search_cropped, target_cropped, &bbox);

      // Unscale the estimation to the real image size.
      BoundingBox bbox_unscaled;
      bbox.Unscale(search_cropped, &bbox_unscaled);

      // Draw white bouding boxes on the groud truth and red bouding box on the estimated location.
      target_bbox.Draw(255, 255, 255, &target_origin);
      search_bbox.Draw(255, 255, 255, &search_origin);
      CropPadImage(target_bbox, target_origin, &target_cropped);
      CropLessPadImage(search_bbox, search_origin, target_cropped.cols, target_cropped.rows, &search_cropped);
      bbox_unscaled.Draw(255, 0, 0, &search_cropped);

      // Show the estimated location
      std::cout << folder_path
          << " " << bbox_unscaled.x1_ << " " << bbox_unscaled.y1_
          << " " << bbox_unscaled.x2_ << " " << bbox_unscaled.y2_
          << std::endl;

      // Write the result images
      cv::imwrite(folder_path + "/target_goturn.jpg", target_cropped);
      cv::imwrite(folder_path + "/search_goturn.jpg", search_cropped);
    }
  }

  // Show all result images.
  for (size_t index = 0; index < folders.size(); index++) {
    // Get the folder name.
    const std::string& folder_path = folders[index];

    // Read images.
    // target_cropped is the cropped target image for tracked object.
    // search_cropped is the cropped search image with estimated location of tracked object.
    cv::Mat target_cropped = cv::imread(folder_path + "/target_goturn.jpg");
    cv::Mat search_cropped = cv::imread(folder_path + "/search_goturn.jpg");

    // Setup the window size.
    const int padding_top = 50;
    const int dstWidth = target_cropped.cols * 2;
    const int dstHeight = target_cropped.rows + padding_top;

    // Put images together
    cv::Mat dst = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat targetROI = dst(cv::Rect(0, padding_top, target_cropped.cols, target_cropped.rows));
    target_cropped.copyTo(targetROI);
    targetROI = dst(cv::Rect(target_cropped.cols, padding_top, target_cropped.cols, target_cropped.rows));
    search_cropped.copyTo(targetROI);

    // Write down the title.
    cv::putText(dst, folder_path, cv::Point(target_cropped.cols / 2, padding_top / 2), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 255, 0), 2.0);

    // Show the result.
    cv::namedWindow("Regressor Test", cv::WINDOW_NORMAL);
    cv::imshow("Regressor Test", dst);
    cv::waitKey(0);
  }

  return 0;
}
