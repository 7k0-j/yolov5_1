#include "DPLPose.h"


// 读模型文件（onnx)
bool YoloPose::readModel(Net& net, string& netPath, bool isCuda) {
	try {
		net = cv::dnn::readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		cv::cuda::setDevice(4);
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

//解析xml
bool YoloPose::ReadMark(std::string& _path, PoseOutput& mk) {
	std::string path = _path;
	tinyxml2::XMLDocument doc;
	auto mpath = path.c_str();
	if (doc.LoadFile(mpath) == XML_ERROR_FILE_NOT_FOUND) {
		std::cout << "不存在标记文件！" << endl;
		return false;
	}
	tinyxml2::XMLElement* root = doc.RootElement();
	for (root = root->FirstChildElement(); root; root = root->NextSiblingElement()) {
		string name = root->Name();
		//读取person标签
		if (name == "person") {
			for (XMLElement* tmp = root->FirstChildElement(); tmp; tmp = tmp->NextSiblingElement()) {
				if (string(tmp->Name()) == "bndbox") {
					mk.box.x = stoi(tmp->FirstChildElement("xmin")->GetText());
					mk.box.y = stoi(tmp->FirstChildElement("ymin")->GetText());
					mk.box.width = stoi(tmp->FirstChildElement("xmax")->GetText()) - mk.box.x;
					mk.box.height = stoi(tmp->FirstChildElement("ymax")->GetText()) - mk.box.y;
				}
				else if (string(tmp->Name()) == "keypoints") {
					string data = tmp->GetText();
					std::istringstream iss(data);
					int x, y, v;
					iss >> x >> y >> v;
					mk.keypoints.emplace_back(x, y, v);
				}
			}
		}
	}
	return true;
}

//写入xml
bool YoloPose::WriteMark(std::string& _path, PoseOutput& mk) {
	std::string path = _path;
	tinyxml2::XMLDocument doc;
	auto mpath = path.c_str();
	if (doc.LoadFile(mpath) == XML_ERROR_FILE_NOT_FOUND) {
		std::cout << "不存在标记文件！" << endl;
		return false;
	}
	tinyxml2::XMLElement* root = doc.RootElement();
	tinyxml2::XMLElement* person = root->FirstChildElement("person");
	tinyxml2::XMLElement* box = person->FirstChildElement("bndbox");
	string xmin = to_string(mk.box.x);
	string xmax = to_string(mk.box.x + mk.box.width - 1);
	string ymin = to_string(mk.box.y);
	string ymax = to_string(mk.box.y + mk.box.height - 1);
	box->FirstChildElement("xmin")->SetText(xmin.c_str());
	box->FirstChildElement("xmax")->SetText(xmax.c_str());
	box->FirstChildElement("ymin")->SetText(ymin.c_str());
	box->FirstChildElement("ymax")->SetText(ymax.c_str());
	doc.SaveFile(mpath);
	return true;
}

//姿态检测
bool YoloPose::PoseDetect(cv::Mat& SrcImg, cv::dnn::Net& net, PoseOutput& output) {
	int cols = SrcImg.cols;
	int rows = SrcImg.rows;
	int maxLen = max(cols, rows);
	Mat netInput = Mat::zeros(maxLen, maxLen, CV_8UC3);
	SrcImg.copyTo(netInput(Rect(0, 0, cols, rows)));
	Mat blob = blobFromImage(netInput, 1 / 255.0, cv::Size(netInputWidth, netInputHeight));
	net.setInput(blob);
	std::vector<cv::Mat> netOutput;
	net.forward(netOutput, net.getUnconnectedOutLayersNames());

	float* pdata = (float*)netOutput[0].data;
	cv::Size netOutputShape = { netOutput[0].size[2],netOutput[0].size[1] };
	cv::Mat all_data(netOutputShape, CV_32FC1, pdata);
	cv::transpose(all_data, all_data);
	double max_conf;
	Point ind;
	minMaxLoc(all_data.col(4), 0, &max_conf, 0, &ind);
	float* result = (float*)all_data.row(ind.y).data;		//取出置信度最高的一行

	float ratio_h = (float)netInput.rows / netInputHeight;
	float ratio_w = (float)netInput.cols / netInputWidth;
	float x = result[0];  //x
	float y = result[1];  //y
	float w = result[2];  //w
	float h = result[3];  //h
	int left = std::max(int((x - 0.5 * w) * ratio_w), 0);
	int top = std::max(int((y - 0.5 * h) * ratio_h), 0);
	int width = std::max(int(w * ratio_w), 0);
	int height = std::max(int(h * ratio_h), 0);
	if (left + width > cols) {
		width = cols - left;
	}
	if (top + height > rows) {
		height = rows - top;
	}
	output.box = Rect(left, top, width, height);
	output.confidence = result[4];
	Keypoint keypoint;
	for (int i = 0; i < (netOutputShape.height - 5) / 3; i++) {
		keypoint.point = Point(result[5 + i * 3] * ratio_w, result[6 + i * 3] * ratio_h);
		keypoint.visible = result[7 + i * 3];
		output.keypoints.push_back(keypoint);
	}
	if (output.keypoints.size())
		return true;
	else
		return false;
}

