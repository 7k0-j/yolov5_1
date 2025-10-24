#include "DPL.h"


// 读模型文件（onnx)
bool Yolo::ReadModel(string& netPath, bool isCuda) {
	try {
		this->net = cv::dnn::readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		cv::cuda::setDevice(1);
		this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	//cpu
	else {
		this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

//解析xml
bool Yolo::ReadMark(std::string& _path, std::vector<cv::Rect>& mk) {
	tinyxml2::XMLDocument doc;
	if (doc.LoadFile(_path.c_str()) == XML_ERROR_FILE_NOT_FOUND) {
		std::cout << "不存在标记文件！" << endl;
		return false;
	}
	tinyxml2::XMLElement* root = doc.RootElement();
	for (root = root->FirstChildElement(); root; root = root->NextSiblingElement()) {
		string name = root->Name();
		//读取object标签
		if (name == "object") {
			cv::Rect t;
			for (XMLElement* tmp = root->FirstChildElement(); tmp; tmp = tmp->NextSiblingElement()) {
				//tmp指向身体部位
				for (XMLElement* tmp2 = tmp->FirstChildElement(); tmp2; tmp2 = tmp2->NextSiblingElement()) {
					//tmp2指向物品类型
					XMLElement* tmp3 = tmp2->FirstChildElement("bndbox");
					t.x = stoi(tmp3->FirstChildElement("xmin")->GetText());
					t.y = stoi(tmp3->FirstChildElement("ymin")->GetText());
					t.width = stoi(tmp3->FirstChildElement("xmax")->GetText()) - t.x;
					t.height = stoi(tmp3->FirstChildElement("ymax")->GetText()) - t.y;
					mk.push_back(t);
				}
			}
		}
	}
	return true;
}

//用模型net检测SrcImg中的目标，结果存在output中
bool Yolo::Detect(Mat& SrcImg, vector<Output>& output) {
	if (SrcImg.empty())
		return false;
	int _rows = SrcImg.rows, _cols = SrcImg.cols;
	int MaxLen = max(_rows, _cols);
	Mat InputImg;
	copyMakeBorder(SrcImg, InputImg, 0, MaxLen - _rows, 0, MaxLen - _cols, cv::BORDER_CONSTANT, 0);
	Mat blob = blobFromImage(InputImg, 1.0 / 255.0, cv::Size(netInputWidth, netInputHeight));
	std::vector<cv::Mat> netOutput;
	this->net_mutex.lock();
	this->net.setInput(blob);
	this->net.forward(netOutput, this->net.getUnconnectedOutLayersNames());
	this->net_mutex.unlock();

	float ratio_h = (float)InputImg.rows / netInputHeight;
	float ratio_w = (float)InputImg.cols / netInputWidth;
	std::vector<int> classIds;					// 结果id数组
	std::vector<float> confidences;				// 结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;				// 每个id矩形框
	float* pdata = (float*)netOutput[0].data;
	int net_height = netOutput[0].size[1];
	int net_width = netOutput[0].size[2];
	for (int r = 0; r < net_height; r++) {
		float box_score = pdata[4];				// 获取每一行的box框中含有某个物体的概率
		if (box_score >= boxThreshold) {
			cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
			max_class_score = (float)max_class_score;
			if (max_class_score >= classThreshold) {
				float x = pdata[0];
				float y = pdata[1];
				float w = pdata[2];
				float h = pdata[3];
				int left = max(int((x - 0.5 * w) * ratio_w + 0.5), 0);
				int top = max(int((y - 0.5 * h) * ratio_h + 0.5), 0);
				int width = int(w * ratio_w + 0.5);
				int height = int(h * ratio_h + 0.5);
				width = max(min(_cols - left, width), 0);
				height = max(min(_rows - top, height), 0);
				classIds.emplace_back(classIdPoint.x);
				confidences.emplace_back(max_class_score * box_score);
				boxes.emplace_back(left, top, width, height);
			}
		}
		pdata += net_width;		// 下一行
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, boxThreshold, nmsThreshold, nms_result);
	for (int idx : nms_result) {
		if (boxes[idx].empty())
			continue;
		Scalar mean(255), stddev(255);
		cv::meanStdDev(SrcImg(boxes[idx]), mean, stddev);
		if (mean[0] < 18 && stddev[0] < 12 && confidences[idx] < 0.7)
			continue;
		output.emplace_back(classIds[idx], confidences[idx], boxes[idx]);
	}
	return !output.empty();
}

//多线程

bool Yolo::Detect_Async2(vector<Mat>& SrcImgs, vector<vector<Output>>& output) {
	if (SrcImgs.empty())
		return false;
	int batch_size = SrcImgs.size(), _rows = SrcImgs[0].rows, _cols = SrcImgs[0].cols, comb_size = 1;
	output.resize(batch_size);
	int tm = (batch_size + comb_size - 1) / comb_size;
	std::thread threads[tm];
	for (int t = 0; t < tm; t++) {
		threads[t] = thread([=, &output]{
			Mat ImgComb;
			if (comb_size <= 4) {
				int MaxLen = max(_rows, comb_size * _cols);
				ImgComb = Mat::zeros(MaxLen, MaxLen, CV_8UC3);
			}
			else {
				int MaxLen = max(2 * _rows, (comb_size + 1) / 2 * _cols);
				ImgComb = Mat::zeros(MaxLen, MaxLen, CV_8UC3);
			}
			for (int i = 0; i < comb_size; i++) {
				int pos_x = i / 2 * _cols, pos_y = i % 2 * _rows, img_id = t * comb_size + i;
				if (img_id >= batch_size)	break;
				if (comb_size <= 4) { pos_x = i * _cols, pos_y = 0; }
				SrcImgs[img_id].copyTo(ImgComb(Rect(pos_x, pos_y, _cols, _rows)));
			}
			vector<Output> result;
			Detect(ImgComb, result);
			//将检测结果对应到每张图像		
			for (auto& i : result) {
				int m = i.box.y / _rows, n = i.box.x / _cols;
				i.box.x -= n * _cols;
				i.box.y -= m * _rows;
				if (comb_size <= 4)
					output[t * comb_size + n].push_back(i);
				else
					output[t * comb_size + m + n * 2].push_back(i);
			}
		});
	}
	for (auto& th : threads) {
		th.join();
	}
	return true;
}


//多线程-线程池

bool Yolo::Detect_Async(vector<Mat>& SrcImgs, vector<vector<Output>>& output) {
	if (SrcImgs.empty())
		return false;
	int batch_size = SrcImgs.size(), _rows = SrcImgs[0].rows, _cols = SrcImgs[0].cols, comb_size = 1;
	output.resize(batch_size);
	int tm = (batch_size + comb_size - 1) / comb_size;
	vector<std::future<bool>> vecfuture(tm);
	vector<Mat> ImgCombs(tm);
	vector<vector<Output>> temp(tm);
	for (int t = 0; t < tm; t++) {
		if (comb_size <= 4) {
			int MaxLen = max(_rows, comb_size * _cols);
			ImgCombs[t] = Mat::zeros(MaxLen, MaxLen, CV_8UC3);
		}
		else {
			int MaxLen = max(2 * _rows, (comb_size + 1) / 2 * _cols);
			ImgCombs[t] = Mat::zeros(MaxLen, MaxLen, CV_8UC3);
		}
		for (int i = 0; i < comb_size; i++) {
			int pos_x = i / 2 * _cols, pos_y = i % 2 * _rows, img_id = t * comb_size + i;
			if (img_id >= batch_size)	break;
			if (comb_size <= 4) { pos_x = i * _cols, pos_y = 0; }
			SrcImgs[img_id].copyTo(ImgCombs[t](Rect(pos_x, pos_y, _cols, _rows)));
		}
		//vecfuture[t] = threadPool.addTask(std::bind(&Yolo::Detect, this, ref(ImgComb), ref(temp[t])));
		vecfuture[t] = threadPool.addTask([this](cv::Mat& SrcImg, vector<Output>& output) {
			return Detect(SrcImg, output);
		}, ref(ImgCombs[t]), ref(temp[t]));
	}
	for (int t = 0; t < tm; t++) {
		vecfuture[t].get();
		//将检测结果对应到每张图像
		for (auto& i : temp[t]) {
			int m = i.box.y / _rows, n = i.box.x / _cols;
			i.box.x -= n * _cols;
			i.box.y -= m * _rows;
			if (comb_size <= 4)
				output[t * comb_size + n].push_back(i);
			else
				output[t * comb_size + m + n * 2].push_back(i);
		}
	}
	return true;
}


// 应对偶发的检测框重叠，用在Detect之后
void Yolo::mergeRect(std::vector<Output>& res) {
	if (res.empty())
		return;
	else {
		int resNum = res.size();
		for (int i = 0; i < resNum - 1; ++i) {
			for (int j = i + 1; j < resNum; ++j) {
				Rect tmp = res[i].box & res[j].box;// 取两框交集，看是否有重叠
				if (tmp.empty()) continue;
				else {
					// res[i]与res[j]重叠
					float minS = min(res[i].box.height * res[i].box.width, res[j].box.height * res[j].box.width);// 较小框面积
					float tmpS = tmp.height * tmp.width;// 交集面积
					if (tmpS / minS <= mergeThreshold) {
						continue;
					}
					else {
						// 重叠面积超过阈值
						// 更新res[i]为并集，更新新res[i]置信度，删除res[j]
						res[i].box = res[i].box | res[j].box;
						res[i].confidence = max(res[i].confidence, res[j].confidence);
						res.erase(res.begin() + j);
						resNum = res.size();// 更新res有效大小
					}
				}
			}
		}
	}
}
