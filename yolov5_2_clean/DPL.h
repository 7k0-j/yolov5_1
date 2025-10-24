#pragma once
#include <opencv2/opencv.hpp>
#include "tinyxml2.h"
#include "ThreadPool.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace tinyxml2;

struct Output {
	int id = 0;							// 结果类别id
	float confidence = 0.0f;			// 结果置信度
	cv::Rect box = { 0, 0, 0, 0 };		// 矩形框
	Output() {}
	Output(int _id, float _conf, cv::Rect _box) : id(_id), confidence(_conf), box(_box) {}
};

class Yolo {
public:
	ThreadPool threadPool;
	Yolo(float Trust = 0.35) {
		boxThreshold = Trust;
		classThreshold = Trust;
		threadPool.startPool();
	}
	~Yolo() {}
	bool ReadModel(std::string& netPath, bool isCuda=true);
	bool ReadMark(std::string& _path, std::vector<cv::Rect>& mk);
	bool Detect(cv::Mat& SrcImg, vector<Output>& output);
	bool Detect_Async(vector<cv::Mat>& SrcImgs, vector<vector<Output>>& output);
	bool Detect_Async2(vector<cv::Mat>& SrcImgs, vector<vector<Output>>& output);
	void mergeRect(std::vector<Output>& res);
	
private:
	cv::dnn::Net net;	// 模型类
	std::mutex net_mutex;
	const int netInputWidth = 1280;   //ONNX图片输入宽度
	const int netInputHeight = 1280;  //ONNX图片输入高度

	float boxThreshold = 0.35;// 置信度低于boxThreshold的扔掉
	float classThreshold = 0.35;
	float nmsThreshold = 0.45;
	float mergeThreshold = 0.5;// 重叠框融合系数，重叠面积/两框中最小框面积，超过这个比例认为存在多个检测框框住了同一个目标

	std::vector<std::string> className = { "obj" };// 二分类，不含其他种类目标，之后可按训练方式追加
};
