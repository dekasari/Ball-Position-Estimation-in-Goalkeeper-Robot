#include <iostream>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include "/home/dekasari/cv-helpers.hpp"
#include <chrono>
#include <ctime>

#define D_COL 3
#define D_ROW 3

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace rs2;

const size_t inWidth      = 300;
const size_t inHeight     = 300;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;
const char* classNames[]  = {"background", "ball"};
const int camHeight       = 580;

int left_ball_coord = 0;
int center_ball_coord = 0;
int right_ball_coord = 0;

int main(int argc, char** argv)
{
    std::chrono::time_point<std::chrono::system_clock> end_time, start_time;
    std::chrono::time_point<std::chrono::system_clock> end_time_s, start_time_s;
    std::chrono::duration<double> elapsed_seconds;
    std::chrono::duration<double> elapsed_seconds_s;
    start_time_s = std::chrono::system_clock::now();

    int counterFPS = 0;
    int cloneCounterFPS = 0;

    //Load DNN Model
    String tf_model_ori = "/home/dekasari/frozen_inference_graph.pb";
    String tf_config_ori = "/home/dekasari/MobilenetV2_SSD.pbtxt";
    Net net = dnn::readNetFromTensorflow(tf_model_ori, tf_config_ori);

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8 , 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 360, RS2_FORMAT_Z16, 60);

    pipeline pipe;
    auto config = pipe.start(cfg);
    auto profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();

    // get param intrinsics
    auto intr = profile.get_intrinsics();
    fprintf(stderr, "fx: %.2f\n", intr.fx);
    rs2::align align_to(RS2_STREAM_COLOR);
   
    namedWindow("Display image", 1);

     //set the callback function for any mouse event
    setMouseCallback("Display image", CallBackFunc, NULL);

    Mat graycloneMat = Mat::zeros(640, 360, CV_8UC1);;
    Mat imgClone;
	Mat gray_box = Mat::zeros(640, 360, CV_8UC1); 
    Mat cropped_box = Mat::zeros(640, 360, CV_8UC1); 
	float pixelValue[D_ROW][D_COL];
	float stddev[D_ROW][D_COL];

    while ((waitKey(1) != 27))
    {
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = static_cast<int>(color_frame.get_frame_number());

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        Mat resized_mat;
        resize(color_mat, resized_mat, Size(inWidth, inHeight));
        auto depth_mat = depth_frame_to_meters(depth_frame);

        Mat inputBlob = blobFromImage(color_mat, inScaleFactor, Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
        // Mat inputBlob = blobFromImage(resized_mat, 1.0, Size(inWidth, inHeight), true, false); //Convert Mat to batch of images

        //Mobilenet
        net.setInput(inputBlob, "image_tensor"); //set the network input
        Mat detection = net.forward("detection_out"); //compute output
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        float confidenceThreshold = 0.80;

        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if(confidence > confidenceThreshold)
            {
                Mat blankFrame(color_mat.size(), color_mat.type(), cv::Scalar(0, 0, 0));
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
                int idx = static_cast<int>(detectionMat.at<float>(i, 1));

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

                Rect object_ori((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(color_mat, object_ori, Scalar(255, 255, 0), 2);

                //COORDINATE EXTRACTION ALGORITHM
				Rect box[D_ROW][D_COL];
				float fw = object_ori.width, fh = object_ori.height;

				// cvtColor(color_mat, graycloneMat, cv::COLOR_RGB2GRAY);
				int customWidth = object_ori.width;
				int customHeight = object_ori.height;
                int customX = object_ori.x;
                int customY = object_ori.y;

                if(customX < 0)
                    customX = 0;
                if(customY < 0)
                    customY = 0;

				if(customX + customWidth >= color_mat.cols){
					customWidth = color_mat.cols - 1 - customX;
				}
				if(customY + customHeight >= color_mat.rows){
					customHeight = color_mat.rows - 1 - customY;
				}

                cropped_box = color_mat(Rect(customX, customY, customWidth, customHeight));
                cvtColor(cropped_box, gray_box, cv::COLOR_RGB2GRAY);

				memset(pixelValue, 0, sizeof(pixelValue));
				memset(stddev, 0, sizeof(stddev));
			
				for(int k = 0; k < D_COL; k++){
					for(int l = 0; l < D_ROW; l++){
						box[l][k].x = fw/D_COL * k + xLeftBottom;
						box[l][k].width = fw/D_COL;
						box[l][k].y = fh/D_ROW * l + yLeftBottom;
						box[l][k].height = fh/D_ROW;

						int count_width = box[l][k].x + box[l][k].width;
						int count_height = box[l][k].y + box[l][k].height;

						for(int m = box[l][k].x; m < count_width; m++){
							for(int n = box[l][k].y; n < count_height; n++){
								pixelValue[l][k] += gray_box.at<uchar>(n, m);
							}
						}

						//mean
						pixelValue[l][k] /= (count_width * count_height);

						//std dev
						for(int m =  box[l][k].x; m < count_width; m++){
							for(int n = box[l][k].y; n < count_height; n++){
								stddev[l][k] += pow((gray_box.at<uchar>(n, m) - pixelValue[l][k]), 2);
							}
						}
						stddev[l][k] /= (count_width * count_height);

						stddev[l][k] = sqrt(stddev[l][k]);
						// fprintf(stderr, "avg: %1.5f std: %1.5f || j: %d i: %d cw : %d ch : %d\n", pixelValue[l][k], stddev[l][k], l, k, count_width, count_height);
					}
				}

				float min = 9999999999;
				int idx_i_min = 0;
				int idx_j_min = 0;

				for(int sdev_i = 0; sdev_i <  D_COL; sdev_i++){
					for(int sdev_j = 0; sdev_j < D_ROW; sdev_j++){
						if(stddev[sdev_j][sdev_i] < min){
							min = stddev[sdev_j][sdev_i];
							idx_i_min = sdev_i;
				 			idx_j_min = sdev_j;
						}
					}
				}

                idx_i_min = 1;
                idx_j_min = 1;

                std::ostringstream ss;
                ss << classNames[objectClass] << " " << confidence;
                String conf(ss.str());

                String prob = conf;
                int baseLine = 0;
                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                Size probSize = getTextSize(prob, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                auto center = (object_ori.br() + object_ori.tl())*0.5;
                rectangle(color_mat, box[idx_j_min][idx_i_min], Scalar(255, 0, 255), -1);
                putText(color_mat, ss.str(), center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
                putText(color_mat, prob, center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

               //BALL POSITION ESTIMATION
               //calculate real world coordinates
                int center_box_x = box[idx_j_min][idx_i_min].x + box[idx_j_min][idx_i_min].width / 2;//(box[idx_j_min][idx_i_min].x + box[idx_j_min][idx_i_min].width) / 2;
                int center_box_y = box[idx_j_min][idx_i_min].y + box[idx_j_min][idx_i_min].height / 2;//(box[idx_j_min][idx_i_min].y + box[idx_j_min][idx_i_min].height) / 2;
                float dist = depth_frame.get_distance(center_box_x, center_box_y) * 1000; //convert to mm
                circle(color_mat, Point(center_box_x, center_box_y), 3, Scalar(0, 0, 255), -1, 8,0);
                circle(color_mat, Point(color_mat.cols/2, color_mat.rows/2), 3, Scalar(255,0,255), -1, 8,0);

                double theta = acos(camHeight / dist); //in rad
                double xTemp, yTemp, ztemp;
                xTemp = dist*(center_box_x -intr.ppx)/intr.fx;
                yTemp = dist*(center_box_y -intr.ppy)/intr.fy;
                ztemp = sqrt(pow(dist, 2) - pow(yTemp, 2));
                yTemp = 570 - yTemp - 85.5;

                // circle(color_mat, Point(center_box_x, center_box_y), 3, Scalar(255,255,255), -1, 8,0);

                fprintf(stderr, "xTemp: %.2f || yTemp: %.2f || ztemp: %.2f || distpow: %.2f\n", xTemp, yTemp, ztemp, dist);
            }
        }
        end_time_s = std::chrono::system_clock::now();
        elapsed_seconds_s = end_time_s - start_time_s;
        counterFPS++;
        
        if(elapsed_seconds_s.count() > 1.0)
        {
            start_time_s = std::chrono::system_clock::now();

            cloneCounterFPS = counterFPS;
            counterFPS = 0;
        }
        std::ostringstream sf;
        sf << "FPS : " << cloneCounterFPS;
        
        putText(color_mat, sf.str(), Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
        // fprintf(stderr, "FPS = %d\n", cloneCounterFPS);

        imshow("Display image", color_mat);
        // imshow("Cropped Gray", gray_box);
        // imshow("cropped_box", cropped_box);
        // imshow("graycloneMat", graycloneMat);
    }

    return 0;
}