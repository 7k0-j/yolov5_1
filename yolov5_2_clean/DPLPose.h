#pragma once
#include <opencv2/opencv.hpp>
#include "tinyxml2.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace tinyxml2;


struct Keypoint {		
	cv::Point point = { 0, 0 };	//关键点的坐标位置
	float visible = 0.0f;		//关键点的可见度(置信度)
	Keypoint(int x = 0, int y = 0, int v = 0) : point(x, y), visible(v) {}
};

struct PoseOutput {
	cv::Rect box = { 0, 0, 0, 0 };		//person类别的矩形框
	float confidence = 0.0f;   //结果置信度
	std::vector<Keypoint> keypoints;	//关键点
};

class YoloPose {		//姿态检测类
public:
	YoloPose() {}
	~YoloPose() {}
	bool readModel(cv::dnn::Net& net, std::string& netPath, bool isCuda=true);
	bool ReadMark(std::string& _path, PoseOutput& mk);
	bool WriteMark(std::string& _path, PoseOutput& mk);
	bool PoseDetect(cv::Mat& SrcImg, cv::dnn::Net& net, PoseOutput& PoseResult);
private:		
	const int netInputWidth = 1280;		//ONNX图片输入宽度
	const int netInputHeight = 1280;		//ONNX图片输入高度
};
