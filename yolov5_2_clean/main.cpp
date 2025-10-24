#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>

#include "DataStatistics.h"
#include "DPL.h"
#include "DPLPose.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
namespace fsys = std::filesystem;

int DetecMOD = 2;// 检测模式:   0:不输出可视化结果  1:调试模式  2:保存检测结果图
float Trust = 0.4;// 置信度阈值
float PoseThreshold = 0.03;
int data_n = 1;// 测试数据集n
int set_num = 2;	// 一组图像的张数: 全回波：2；多平面：4
int batch_size = 8;
std::string workplace = "/home/star/WorkSpace/qkl/yolov5_2/";// 工作路径
string ModelName = "v6.2_zhengzhou";
//string PoseModelName = "v5.6";
//string DetectModelPath = "/home/star/WorkSpace/jdq/yolov5/model/detect/V5.30/DPL.onnx";// 目标检测模型地址
string DetectModelPath = "/home/star/WorkSpace/qkl/yolov5_2/model/detect/v6.2_zhengzhou/best.onnx";
//string PoseModelPath = workplace + "model/pose/" + PoseModelName + "/s_model_best.onnx";// 姿态检测模型地址


// 姿态检测测试

// int main() {
// 	vector<string> dataset = { "24_4_25", "24_1_12", "24_1_4", "23_12_20" };
//  	//string ImgROOT = "/media/data/PoseDataset/全回波_new/" + dataset[data_n-1] + "/"; //数据集根目录
// 	string ImgROOT = "/home/star/WorkSpace/Dataset/dataset2/PoseDataset/train/bmpimages/bmp_24_08_13/HWT多平面全回波训练库_调整站姿_46（共计8000张）20240809/"; //数据集根目录
// 	string outPath = workplace + "runs/pose/" + PoseModelName + "/"; //输出结果的保存位置
// 	fsys::create_directories(outPath);
// 	YoloPose DPLPose;
// 	Net net;// 模型类实例化
// 	if (DPLPose.readModel(net, PoseModelPath)) {// 读模型
// 		std::cout << "Read Net : " << fsys::path(PoseModelPath) << "\n"
// 			<< "数据集 : " << fsys::path(ImgROOT).parent_path() << endl;
// 	}
// 	else {
// 		std::cout << "\033[1;31mread model fail!\033[0m" << endl;
// 		return 0;
// 	}
// 	vector<int> res(3,0);//保存检出结果数目: 检出数、误检数、总数
// 	vector<string> dir = { "." };
// 	for (int a = 0; a < dir.size(); ++a) {
// 		res.assign(res.size(), 0);	//将res清零
// 		string tmpPath = ImgROOT + "原始库/";//缓存路径，图像文件夹所在目录
// 		cv::String img_Folder;		
// 		// 数据库测试模式
// 		img_Folder = tmpPath + "*.*";
// 		std::vector<cv::String> img_Filenames;
// 		cv::glob(img_Folder, img_Filenames);
// 		std::sort(img_Filenames.begin(), img_Filenames.end());
// 		clock_t start = clock();
// 		for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {	
// 			tqdm(img_Count + 1, img_Filenames.size());
// 			// 读取当前样本对应的标记
// 			string imgname = fsys::path(img_Filenames[img_Count]).filename().string();
// 			string markPath = ImgROOT + "标注库/" + imgname.replace(imgname.size() - 4, 4, ".xml");
// 			PoseOutput mark;
// 			if (!DPLPose.ReadMark(markPath, mark)) {
// 				std::cout << "标记读取错误" << endl;
// 				continue;
// 			}// 表示该样本没有对应的标记文件，跳到下一张
// 			Mat imgInitial = imread(img_Filenames[img_Count]);
// 			PoseOutput PoseResult;
// 			DPLPose.PoseDetect(imgInitial, net, PoseResult);// 姿态检测
// 			//计算检出率
// 			vector<int> markCmp = ComparePoseMark(PoseResult, mark, res, PoseThreshold);
// 			if (DetecMOD == 0) {		//仅测试
// 				continue;
// 			}
// 			//可视化结果
// 			cv::rectangle(imgInitial, mark.box, Scalar(255,0,0), 3);
// 			for (auto i : mark.keypoints) {
// 				cv::circle(imgInitial, i.point, 7, Scalar(255,0,0), cv::FILLED);
// 			}
// 			string cofi = to_string(int(round(PoseResult.confidence * 10000)));
// 			cofi.insert(cofi.length() - 2, "."); // 插入小数点
// 			cv::putText(imgInitial, cofi, Point(PoseResult.box.x - 15, PoseResult.box.y - 15), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 2, 0.3);
// 			cv::rectangle(imgInitial, PoseResult.box, Scalar(0,0,255), 2);
// 			for (auto i : PoseResult.keypoints) {
// 				//if (i.visible>0.5)
// 				cv::circle(imgInitial, i.point, 5, Scalar(0, 0, 255),cv::FILLED);
// 			}
// 			if (DetecMOD == 1) {		//仅调试，不保存结果
// 				continue;
// 			}
// 			// 保存结果
// 			else if (DetecMOD == 2) {
// 				vector<string> outRecFilePath = {"all"};
// 				if(!markCmp.empty())
// 					outRecFilePath.emplace_back("error");
// 				// else
// 				//  	outRecFilePath.emplace_back("right");
// 				if (!outRecFilePath.empty()) {
// 					string filename = fsys::path(img_Filenames[img_Count]).filename().string();
// 					filename.replace(filename.size() - 4, 4, ".jpg");
// 					for (string s : outRecFilePath) {
// 						string savePath = outPath + s;
// 						fsys::create_directories(savePath);
// 						imwrite(savePath + "/" + filename, imgInitial);
// 					}
// 				}						
// 			}
// 			else
// 				continue;
// 		}
// 		clock_t end = clock();
// 		std::time_t currentTime = std::time(nullptr);
// 		std::tm localtime;
// 		localtime_r(&currentTime, &localtime);
// 		string date_str = to_string(localtime.tm_mon + 1) + "_" + to_string(localtime.tm_mday);
// 		std::ofstream outFile(outPath + date_str + ".log", std::ios::app);	//以添加的方式打开
// 		if (!outFile.is_open())
// 			cout << "\033[1;31m未能打开: " << outPath << "\033[0m" << endl;
// 		else{
// 			cout << "输出结果记录在 : " << outPath + date_str + ".log" << endl;
// 			outFile << "\n" << "Read Net : " << fsys::path(PoseModelPath) << "\n"
// 				<< "数据集 : " << fsys::path(ImgROOT).parent_path() << "\n"
// 				<< "图像数量 : " << img_Filenames.size() << "\n"
// 				<< "单张图像检测时间: " << (double)(end - start) / (img_Filenames.size() * CLOCKS_PER_SEC) << "\n"
// 				<< "总关键点数: " << res[2] << "\n"
// 				<< "检出数量: " << res[0] << "  检出率: " << 100 * (double)res[0] / (double)res[2] << "%" << "\n"
// 				<< "误检数量: " << res[1] << "  误检率: " << 100 * (double)res[1] / (double)res[2] << "%" << endl;
// 			outFile.close();
// 		}
// 		std::cout << "图像数量 : " << img_Filenames.size() << "\n"
// 			<< "单张图像检测时间: " << (double)(end - start) / (img_Filenames.size() * CLOCKS_PER_SEC) << "\n"
// 			<< "总关键点数: " << res[2] << "\n"
// 			<< "检出数量: " << res[0] << "  检出率: " << 100 * (double)res[0] / (double)res[2] << "%" << "\n"
// 			<< "误检数量: " << res[1] << "  误检率: " << 100 * (double)res[1] / (double)res[2] << "%" << endl;
// 	}
// 	return 0;
// }


//多张拼接检测

// int main() {
// 	vector<string> dataset = { "24_5_19", "24_5_14", "24_5_1", "24_4_22", "24_4_16", "24_4_1/全回波_new" };
//  	//string ImgROOT = "/media/data/dataset3/MPB测试集/" + dataset[data_n-1] + "/"; //数据集根目录
//  	string ImgROOT = "/media/data/民航摸底检出测试/误报高/女3（手腕占比高）/冬/";
// 	// 加载数据集与模型
// 	cv::String img_Folder = ImgROOT + "原始库/*.*";
// 	std::vector<cv::String> img_Filenames;
// 	cv::glob(img_Folder, img_Filenames);
// 	if (img_Filenames.empty()) {
// 		std::cout << "\033[1;31m数据集有误!\033[0m" << endl;
// 		return 0;
// 	}
// 	std::sort(img_Filenames.begin(), img_Filenames.end());
// 	if (!fsys::exists(DetectModelPath)) {
// 		std::cout << "\033[1;31m模型路径有误!\033[0m" << endl;
// 		return 0;
// 	}
// 	Yolo DPL(Trust);// 检测相关函数类实例化	
// 	if (!DPL.ReadModel(DetectModelPath)) {// 读模型
// 		std::cout << "\033[1;31mFailed to Read Model!\033[0m" << endl;
// 		return 0;
// 	}
// 	std::cout << "Read Net : " << fsys::path(DetectModelPath) << "\n" << "数据集 : " << fsys::path(ImgROOT) << "\n"
// 		<< "置信度 : " << Trust << "\t\t batch_size : " << batch_size << endl;
// 	string outPath = workplace + "runs/detect/" + ModelName + "/"; // 输出结果的保存位置
// 	fsys::create_directories(outPath);
// 	vector<string> imagePaths;
// 	vector<vector<cv::Rect>> marks;
// 	vector<Mat> ImgOris;
// 	vector<vector<Output>> DPLresult;	// Output:Rect框、float置信度、int类别
// 	vector<vector<bool>> markCmp;
// 	imagePaths.reserve(batch_size);
// 	marks.reserve(batch_size);
// 	ImgOris.reserve(batch_size);
// 	DPLresult.reserve(batch_size);
// 	markCmp.reserve(batch_size);
// 	vector<int> res(7, 0);	// 保存检出结果：检出数、漏检数、误检数、总标记数、误检次数、违禁品总数、误检违禁品数
// 	vector<long> times = { 0, LONG_MAX, 0 };	// 单批次最长用时、单批次最短用时、总用时
// 	std::chrono::time_point start = std::chrono::system_clock::now();
// 	for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {
// 		tqdm(img_Count + 1, img_Filenames.size(), start);
// 		imagePaths.push_back(img_Filenames[img_Count]);
// 		if (imagePaths.size() < batch_size && img_Count != img_Filenames.size() - 1)
// 			continue;
// 		std::chrono::time_point _start = std::chrono::system_clock::now();
// 		int bm = imagePaths.size();
// 		marks.resize(bm);
// 		ImgOris.resize(bm);
// 		DPLresult.resize(bm);
// 		markCmp.resize(bm);
// 		for (int i = 0; i < bm; i++) {
// 			string imgname = fsys::path(imagePaths[i]).filename().string();
// 			string markPath = ImgROOT + "标注库/" + imgname.substr(0, imgname.size() - 4) + ".xml";
// 			if (!DPL.ReadMark(markPath, marks[i]))
// 				std::cout << "标签读取错误:" << markPath << endl;
// 			ImgOris[i] = imread(imagePaths[i]);
// 		}	
// 		DPL.Detect_Async2(ImgOris, DPLresult);	// 检测
// 		//DPL.mergeRect(DPLresult);	// 检测框后处理：根据最小框重叠面积整合重叠框
// 		std::chrono::time_point _end = std::chrono::system_clock::now();
// 		auto _used_time = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / bm;
// 		cout << "\n" << img_Count << " used_time: " << _used_time << " ms\r\033[A" << flush;
// 		if (_used_time > times[0] && img_Count > batch_size)	times[0] = _used_time;
// 		if (_used_time < times[1])								times[1] = _used_time;
// 		// 统计检出、漏检以及误检		
// 		bool error = false;
// 		for (int i = 0; i < bm; i++) {
// 			markCmp[i] = CompareMark(DPLresult[i], marks[i], res);
// 			if (!error)	error = markCmp[i][1];
// 			if ((i + 1) % set_num == 0) {	// 是一组的最后
// 				// 解析违禁品个数
// 				string imgname = fsys::path(imagePaths[i]).filename().string();
// 				int j = imgname.size() - 9, count = 0;
// 				while (!isalpha(imgname[j])) {
// 					if (imgname[--j] == '_')	count++;
// 				}
// 				res[5] += (count + 2) / 3;	// 更新总违禁品数
// 				if (error) {	// 出现误检
// 					res[4]++;	// 更新误检次数
// 					res[6] += (count + 2) / 3;	// 更新误检违禁品数
// 					error = false;
// 				}
// 			}
// 		}
// 		if (DetecMOD == 0) {
// 			imagePaths.clear();
// 			marks.clear();
// 			ImgOris.clear();
// 			DPLresult.clear();
// 			markCmp.clear();
// 			continue;// 不输出可视化结果，当前图像任务已结束
// 		}
// 		// 将标签框和检出框画到原图上,在imagewatch上显示
// 		for (int i = 0; i < bm; i++) {
// 			for (int j = 0; j < marks[i].size(); j++) {
// 				cv::rectangle(ImgOris[i], marks[i][j], Scalar(255, 0, 0), 5);
// 			}
// 			for (int j = 0; j < DPLresult[i].size(); j++) {
// 				cv::rectangle(ImgOris[i], DPLresult[i][j].box, Scalar(0, 0, 255), 3);
// 			}
// 		}
// 		if (DetecMOD == 1) {		//仅调试，不保存结果
// 			imagePaths.clear();
// 			marks.clear();
// 			ImgOris.clear();
// 			DPLresult.clear();
// 			markCmp.clear();
// 			continue;
// 		}
// 		// 保存结果
// 		else if (DetecMOD == 2) {
// 			vector<string> outRecFilePath;
// 			outRecFilePath.reserve(4);
// 			for (int i = 0; i < bm; i++) {
// 				//outRecFilePath.emplace_back("allRec");
// 				if (markCmp[i][0]) {	//出现漏检
// 					outRecFilePath.emplace_back("missRec");
// 				}
// 				if (markCmp[i][1]) {	//出现误检
// 					outRecFilePath.emplace_back("errorRec");
// 				}
// 				// if (outRecFilePath.empty()) {
// 				// 	outRecFilePath.emplace_back("rightRec");
// 				// }
// 				if (outRecFilePath.empty())
// 					continue;
// 				string imgname = fsys::path(imagePaths[i]).filename().string();
// 				string filename = imgname.substr(0, imgname.size() - 4) + ".jpg";
// 				for (string s : outRecFilePath) {
// 					string savePath = outPath + to_string(Trust).substr(0, 4) + "/" + s;
// 					fsys::create_directories(savePath);
// 					imwrite(savePath + "/" + filename, ImgOris[i]);
// 				}
// 				outRecFilePath.clear();
// 			}
// 		}
// 		imagePaths.clear();
// 		marks.clear();
// 		ImgOris.clear();
// 		DPLresult.clear();
// 		markCmp.clear();
// 	}
// 	std::chrono::time_point end = std::chrono::system_clock::now();
// 	times[2] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
// 	std::time_t currentTime = std::time(nullptr);
// 	std::tm localTime;
// 	localtime_r(&currentTime, &localTime);
// 	string date_str = to_string(localTime.tm_mon + 1) + "_" + to_string(localTime.tm_mday);
// 	std::ofstream outFile(outPath + date_str + ".log", std::ios::app);	//以添加的方式打开
// 	if (!outFile.is_open())
// 		cout << "\33[1;31m未能打开 : " << outPath + date_str + ".log\33[0m" << endl;
// 	else {
// 		cout << "输出结果记录在 : " << outPath + date_str + ".log" << endl;
// 		outFile << "\n" << "Read Net : " << fsys::path(DetectModelPath) << "\n"
// 			<< "数据集 : " << fsys::path(ImgROOT) << "\n"
// 			<< "图像数量 : " << img_Filenames.size() << "\t\t 置信度 : " << Trust << "\t\t batch_size : " << batch_size << "\n"
// 			<< "检测用时: " << times[2] / 60000 << "min " << times[2] % 60000 / 1000 << "s (total)  "
// 				<< times[0] << "ms (max)  " << times[1] << "ms (min)  "
// 				<< times[2] / img_Filenames.size() << "ms (average)" << "\n"
// 			<< "总目标数: " << res[3] << "  总扫描次数: " << img_Filenames.size() / set_num << "\n"
// 			<< "检出数量: " << res[0] << "  检出率: " << 100.0 * res[0] / res[3] << "%" << "\n"
// 			<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "\n"
// 			<< "误检次数: " << res[4] << "  误检率1: " << 100.0 * res[4] / (img_Filenames.size() / set_num) << "%" << "\n"
// 			//<< "误检率3: " << 100.0 * res[2] / (res[0] + res[2]) << "%" << "\n"
// 			<< "违禁品总数: " << res[5] << "  误报违禁品数: " << res[6] << "  误检率2: " << 100.0 * res[6] / res[5] << "%" << endl;
// 		outFile.close();
// 	}
// 	std::cout << "图像数量 : " << img_Filenames.size() << "\n"
// 		<< "检测用时: " << times[2] / 60000 << "min " << times[2] % 60000 / 1000 << "s (total)  "
// 				<< times[0] << "ms (max)  " << times[1] << "ms (min)  "
// 				<< times[2] / img_Filenames.size() << "ms (average)" << "\n"
// 		<< "总目标数: " << res[3] << "  总扫描次数: " << img_Filenames.size() / set_num << "\n"
// 		<< "检出数量: " << res[0] << "  检出率: " << 100.0 * res[0] / res[3] << "%" << "\n"
// 		<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "\n"
// 		<< "误检次数: " << res[4] << "  误检率1: " << 100.0 * res[4] / (img_Filenames.size() / set_num) << "%" << "\n"
// 		//<< "误检率3: " << 100.0 * res[2] / (res[0] + res[2]) << "%" << "\n"
// 		<< "违禁品总数: " << res[5] << "  误报违禁品数: " << res[6] << "  误检率2: " << 100.0 * res[6] / res[5] << "%" << endl;
// 	return 0;
// }


//多张拼接检测-线程池

// #include <unistd.h> // 包含 getpid() 函数所在的头文件
// int main() {
// 	cout << "当前进程PID: " << getpid() << endl;
// 	vector<string> dataset = { "24_5_29", "锐化测试集_24_5_26", "24_5_19", "24_5_14", "24_5_1", "24_4_22", "24_4_16", "24_4_1/全回波_new" };
//  	string ImgROOT = "/home/star/WorkSpace/Dataset/dataset3/" + dataset[data_n-1] + "/"; //数据集根目录
//  	//string ImgROOT = "/home/star/WorkSpace/hanz/C++/CompanyShow/demo/";
// 	// 加载数据集与模型
// 	cv::String img_Folder = ImgROOT + "原始库/*.*";
// 	std::vector<cv::String> img_Filenames;
// 	cv::glob(img_Folder, img_Filenames);
// 	if (img_Filenames.empty()) {
// 		std::cout << "\033[1;31m数据集有误!\033[0m" << endl;
// 		return 0;
// 	}
// 	std::sort(img_Filenames.begin(), img_Filenames.end());
// 	if (!fsys::exists(DetectModelPath)) {
// 		std::cout << "\033[1;31m模型路径有误!\033[0m" << endl;
// 		return 0;
// 	}
// 	Yolo DPL(Trust);// 检测相关函数类实例化	
// 	if (!DPL.ReadModel(DetectModelPath)) {// 读模型
// 		std::cout << "\033[1;31mFailed to Read Model!\033[0m" << endl;
// 		return 0;
// 	}
// 	std::cout << "Read Net : " << fsys::path(DetectModelPath) << "\n" << "数据集 : " << fsys::path(ImgROOT) << "\n"
// 		<< "置信度 : " << Trust << "\t\t batch_size : " << batch_size << endl;
// 	string outPath = workplace + "runs/detect/" + ModelName + "/"; // 输出结果的保存位置
// 	fsys::create_directories(outPath);
// 	vector<string> imagePaths;
// 	vector<vector<cv::Rect>> marks;
// 	vector<Mat> ImgOris;
// 	vector<vector<Output>> DPLresult;	// Output:Rect框、float置信度、int类别
// 	vector<vector<bool>> markCmp;
// 	imagePaths.reserve(batch_size);
// 	marks.reserve(batch_size);
// 	ImgOris.reserve(batch_size);
// 	DPLresult.reserve(batch_size);
// 	markCmp.reserve(batch_size);
// 	vector<int> res(7, 0);	// 保存检出结果：检出数、漏检数、误检数、总标记数、误检次数、违禁品总数、误检违禁品数
// 	vector<long> times = { 0, LONG_MAX, 0 };	// 单批次最长用时、单批次最短用时、总用时
// 	std::chrono::time_point start = std::chrono::system_clock::now();
// 	for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {
// 		tqdm(img_Count + 1, img_Filenames.size(), start);
// 		imagePaths.push_back(img_Filenames[img_Count]);
// 		if (imagePaths.size() < batch_size && img_Count != img_Filenames.size() - 1)
// 			continue;
// 		std::chrono::time_point _start = std::chrono::system_clock::now();
// 		int bm = imagePaths.size();
// 		marks.resize(bm);
// 		ImgOris.resize(bm);
// 		DPLresult.resize(bm);
// 		markCmp.resize(bm);
//         vector<std::future<bool>> markfuture(bm);
// 		for (int i = 0; i < bm; i++) {
// 			string imgname = fsys::path(imagePaths[i]).filename().string();
// 			string markPath = ImgROOT + "标注库/" + imgname.substr(0, imgname.size() - 4) + ".xml";
//             markfuture[i] = DPL.threadPool.addTask([&DPL](std::string& _path, std::vector<cv::Rect>& mk) {
//     			return DPL.ReadMark(_path, mk);
// 			}, ref(markPath), ref(marks[i]));
// 			ImgOris[i] = imread(imagePaths[i]);
// 		}
// 		for (int i = 0; i < bm; i++) {
// 			if (!markfuture[i].get())
// 				std::cout << "标签读取错误:" << imagePaths[i] << endl;
// 		}
// 		DPL.Detect_Async(ImgOris, DPLresult);	// 检测
// 		//DPL.mergeRect(DPLresult);	// 检测框后处理：根据最小框重叠面积整合重叠框
// 		std::chrono::time_point _end = std::chrono::system_clock::now();
// 		auto _used_time = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / bm;
// 		cout << "\n" << img_Count << " 单张检测用时: " << _used_time << " ms  \r\033[A" << flush;
// 		if (_used_time > times[0] && img_Count > batch_size)	times[0] = _used_time;
// 		if (_used_time < times[1])								times[1] = _used_time;
// 		// 统计检出、漏检以及误检
// 		vector<std::future<vector<bool>>> Cmpfuture(bm);
// 		for (int i = 0; i < bm; i++) {
// 			Cmpfuture[i] = DPL.threadPool.addTask(CompareMark, ref(DPLresult[i]), ref(marks[i]), ref(res));
// 		}
// 		bool error = false;
// 		for (int i = 0; i < bm; i++) {
// 			markCmp[i] = Cmpfuture[i].get();
// 			if (!error)	error = markCmp[i][1];
// 			if ((i + 1) % set_num == 0) {	// 是一组的最后
// 				// 解析违禁品个数
// 				string imgname = fsys::path(imagePaths[i]).filename().string();
// 				int j = imgname.size() - 15, count = 0;
// 				while (!isalpha(imgname[j])) {
// 					if (imgname[--j] == '_')	count++;
// 				}
// 				res[5] += (count + 2) / 3;	// 更新总违禁品数
// 				if (error) {	// 出现误检
// 					res[4]++;	// 更新误检次数
// 					res[6] += (count + 2) / 3;	// 更新误检违禁品数
// 					error = false;
// 				}
// 			}
// 		}
// 		if (DetecMOD == 0) {
// 			imagePaths.clear();
// 			marks.clear();
// 			ImgOris.clear();
// 			DPLresult.clear();
// 			markCmp.clear();
// 			continue;// 不输出可视化结果，当前图像任务已结束
// 		}
// 		// 将标签框和检出框画到原图上,在imagewatch上显示
// 		for (int i = 0; i < bm; i++) {
// 			for (int j = 0; j < marks[i].size(); j++) {
// 				cv::rectangle(ImgOris[i], marks[i][j], Scalar(255, 0, 0), 5);
// 			}
// 			for (int j = 0; j < DPLresult[i].size(); j++) {
// 				cv::rectangle(ImgOris[i], DPLresult[i][j].box, Scalar(0, 0, 255), 3);
// 				string cofi = to_string(DPLresult[i][j].confidence).substr(0, 4);
// 				Point position = {DPLresult[i][j].box.x - 15, DPLresult[i][j].box.y - 15};
// 				cv::putText(ImgOris[i], cofi, position, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 0.3);
// 			}
// 		}
// 		if (DetecMOD == 1) {		//仅调试，不保存结果
// 			imagePaths.clear();
// 			marks.clear();
// 			ImgOris.clear();
// 			DPLresult.clear();
// 			markCmp.clear();
// 			continue;
// 		}
// 		// 保存结果
// 		else if (DetecMOD == 2) {
// 			vector<string> outRecFilePath;
// 			outRecFilePath.reserve(4);
// 			for (int i = 0; i < bm; i++) {
// 				outRecFilePath.emplace_back("allRec");
// 				// if (markCmp[i][0]) {	//出现漏检
// 				// 	outRecFilePath.emplace_back("missRec");
// 				// }
// 				// if (markCmp[i][1]) {	//出现误检
// 				// 	outRecFilePath.emplace_back("errorRec");
// 				// }
// 				// if (outRecFilePath.empty()) {
// 				// 	outRecFilePath.emplace_back("rightRec");
// 				// }
// 				if (outRecFilePath.empty())
// 					continue;
// 				string imgname = fsys::path(imagePaths[i]).filename().string();
// 				string filename = imgname.substr(0, imgname.size() - 4) + ".jpg";
// 				for (string s : outRecFilePath) {
// 					string savePath = outPath + to_string(Trust).substr(0, 4) + "/" + s;
// 					fsys::create_directories(savePath);
// 					imwrite(savePath + "/" + filename, ImgOris[i]);
// 				}
// 				outRecFilePath.clear();
// 			}
// 		}
// 		imagePaths.clear();
// 		marks.clear();
// 		ImgOris.clear();
// 		DPLresult.clear();
// 		markCmp.clear();
// 	}
// 	std::chrono::time_point end = std::chrono::system_clock::now();
// 	times[2] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
// 	std::time_t currentTime = std::time(nullptr);
// 	std::tm localTime;
// 	localtime_r(&currentTime, &localTime);
// 	string date_str = to_string(localTime.tm_mon + 1) + "_" + to_string(localTime.tm_mday);
// 	std::ofstream outFile(outPath + date_str + ".log", std::ios::app);	//以添加的方式打开
// 	if (!outFile.is_open())
// 		cout << "\33[1;31m未能打开 : " << outPath + date_str + ".log\33[0m" << endl;
// 	else {
// 		cout << "输出结果记录在 : " << outPath + date_str + ".log" << endl;
// 		outFile << "\n" << "Read Net : " << fsys::path(DetectModelPath) << "\n"
// 			<< "数据集 : " << fsys::path(ImgROOT) << "\n"
// 			<< "图像数量 : " << img_Filenames.size() << "\t\t 置信度 : " << Trust << "\t\t batch_size : " << batch_size << "\n"
// 			<< "检测用时: " << times[2] / 60000 << "min " << times[2] % 60000 / 1000 << "s (total)  "
// 				<< times[0] << "ms (max)  " << times[1] << "ms (min)  "
// 				<< times[2] / img_Filenames.size() << "ms (average)" << "\n"
// 			<< "总目标数: " << res[3] << "  总扫描次数: " << img_Filenames.size() / set_num << "\n"
// 			<< "检出数量: " << res[0] << "  检出率: " << 100.0 * res[0] / res[3] << "%" << "\n"
// 			<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "\n"
// 			<< "误检次数: " << res[4] << "  误检率1: " << 100.0 * res[4] / (img_Filenames.size() / set_num) << "%" << "\n"
// 			//<< "误检率3: " << 100.0 * res[2] / (res[0] + res[2]) << "%" << "\n"
// 			<< "违禁品总数: " << res[5] << "  误报违禁品数: " << res[6] << "  误检率2: " << 100.0 * res[6] / res[5] << "%" << endl;
// 		outFile.close();
// 	}
// 	std::cout << "图像数量 : " << img_Filenames.size() << "\n"
// 		<< "检测用时: " << times[2] / 60000 << "min " << times[2] % 60000 / 1000 << "s (total)  "
// 				<< times[0] << "ms (max)  " << times[1] << "ms (min)  "
// 				<< times[2] / img_Filenames.size() << "ms (average)" << "\n"
// 		<< "总目标数: " << res[3] << "  总扫描次数: " << img_Filenames.size() / set_num << "\n"
// 		<< "检出数量: " << res[0] << "  检出率: " << 100.0 * res[0] / res[3] << "%" << "\n"
// 		<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "\n"
// 		<< "误检次数: " << res[4] << "  误检率1: " << 100.0 * res[4] / (img_Filenames.size() / set_num) << "%" << "\n"
// 		//<< "误检率3: " << 100.0 * res[2] / (res[0] + res[2]) << "%" << "\n"
// 		<< "违禁品总数: " << res[5] << "  误报违禁品数: " << res[6] << "  误检率2: " << 100.0 * res[6] / res[5] << "%" << endl;
// 	return 0;
// }


// 全回波对抗集测试

int main() {
	//vector<string> dataset = { "旧图测试库", "24_9_24", "24_9_10", "24_9_6", "24_8_31", "24_8_26", "24_7_10", "24_6_29" };
 	//string ImgROOT = "/home/star/WorkSpace/Dataset/dataset3/" + dataset[data_n-1] + "/"; //数据集根目录
	//string ImgROOT = "/home/star/WorkSpace/Dataset/dataset3/yuantuku/";
	string ImgROOT = "/home/star/WorkSpace/qkl/DataSet/25_7_2/";
	// 加载数据集与模型
	cv::String img_Folder = ImgROOT + "原始库/*.*";
	//cv::String img_Folder = ImgROOT + "images/*.*";
	std::vector<cv::String> img_Filenames;
	cv::glob(img_Folder, img_Filenames);
	if (img_Filenames.empty()) {
		std::cout << "\033[1;31m数据集有误!\033[0m" << endl;
		return 0;
	}
	std::sort(img_Filenames.begin(), img_Filenames.end());
	Yolo DPL(Trust);// 检测相关函数类实例化
	if (!DPL.ReadModel(DetectModelPath)) {// 读模型
		std::cout << "\033[1;31mFailed to Read Model!\033[0m" << endl;
		return 0;
	}
	std::cout << "Read Net : " << fsys::path(DetectModelPath) << "\n"
		<< "数据集 : " << fsys::path(ImgROOT) << "\n" << "置信度 : " << Trust << endl;
	string outPath = workplace + "runs/detect/" + ModelName + "/"; // 输出结果的保存位置
	fsys::create_directories(outPath);
	vector<int> res(8, 0);	// 保存检出结果：检出数、漏检数、误检数、总标记数、误检数备份、误检次数、违禁品总数、误检违禁品数
	vector<long> times = { 0, LONG_MAX, 0 };	// 单批次最长用时、单批次最短用时、总用时
    std::chrono::time_point start = std::chrono::system_clock::now();
	for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {	
		tqdm(img_Count + 1, img_Filenames.size(), start);
		// 读取当前样本对应的标记
		const string imgname = fsys::path(img_Filenames[img_Count]).filename().string();
		string markPath = ImgROOT + "标注库/" + imgname.substr(0, imgname.size() - 4) + ".xml";
		//string markPath = ImgROOT + "Annotations/" + imgname.substr(0, imgname.size() - 4) + ".xml";
		vector<cv::Rect> mark;
		if (!(DPL.ReadMark(markPath, mark))) {
			std::cout << "标记读取错误" << endl;
			//continue;
		}// 表示该样本没有对应的标记文件，跳到下一张
		Mat imgOri = imread(img_Filenames[img_Count]);
		vector<Output> DPLresult;
		std::chrono::time_point _start = std::chrono::system_clock::now();
		DPL.Detect(imgOri, DPLresult);// 检测
		// 检测框后处理: 根据最小框重叠面积整合重叠框
		//DPL.mergeRect(DPLresult);
		std::chrono::time_point _end = std::chrono::system_clock::now();
		auto _used_time = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
		cout << "\n" << img_Count << " used_time: " << _used_time << "ms  \r\33[A" << flush;
		if (_used_time > times[0] && img_Count > batch_size)	times[0] = _used_time;
		if (_used_time < times[1])								times[1] = _used_time;
		vector<bool> markCmp = CompareMark(DPLresult, mark, res);
		if ((img_Count + 1) % 2 == 0) { // 一组图像检测完成
            // 解析违禁品个数
            int j = imgname.size() - 9, count = 0;
		    while (j > 25 && !isalpha(imgname[j])) {
                if (imgname[--j] == '_') {
                    count++;
                }
            }
            res[6] += (count + 2) / 3;
		    if (res[4] < res[2]) {	// 出现误检			
			    res[5]++;	// 更新误检次数
			    res[4] = res[2];
                res[7] += (count + 2) / 3;	// 更新误检违禁品数	
		    }
		}
		if (DetecMOD == 0)
			continue;// 不输出可视化结果，当前图像任务已结束
		// 调试模式: 在原图上框出检测结果，在imagewatch上显示	
		for (const auto& i : mark) {
			cv::rectangle(imgOri, i, Scalar(255, 0, 0), 3);
		}
		for (auto& i : DPLresult) {
			string cofi = to_string(int(round(i.confidence * 10000)));
			cofi.insert(cofi.length() - 2, "."); // 插入小数点
			cv::putText(imgOri, cofi, Point(i.box.x - 15, i.box.y - 15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 0.3);
			cv::rectangle(imgOri, i.box, Scalar(0,0,255), 2);
		}
		if (DetecMOD == 1) {		//仅调试，不保存结果
			continue;
		}
		// 保存结果
		else if (DetecMOD == 2) {
			vector<string> outRecFilePath;
			if (markCmp[0]) {	// 出现漏检
				outRecFilePath.emplace_back("missRec");		
			}				
			if (markCmp[1]) {	// 出现误检
				outRecFilePath.emplace_back("errorRec");
			}	
		  //if (outRecFilePath.empty()) {	//既没漏检，也没误检
				//outRecFilePath.emplace_back("rightRec");
			//}
			if (!outRecFilePath.empty()) {
				string filename = imgname.substr(0, imgname.size() - 4) + ".jpg";
				for (string s : outRecFilePath) {
					string savePath = outPath + to_string(Trust).substr(0, 4) + "/" + s;
					fsys::create_directories(savePath);
					imwrite(savePath + "/" + filename, imgOri);
				}
			}					
		}
		else
			continue;
	}
	std::chrono::time_point end = std::chrono::system_clock::now();
	times[2] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::time_t currentTime = std::time(nullptr);
	std::tm localtime;
	localtime_r(&currentTime, &localtime);
	string date_str = to_string(localtime.tm_mon + 1) + "_" + to_string(localtime.tm_mday);
	std::ofstream outFile(outPath + date_str + ".log", std::ios::app);	//以添加的方式打开
	if (!outFile.is_open())
		cout << "未能打开 : " << outPath + date_str + ".log" << endl;
	else {
		cout << "输出结果记录在 : " << outPath + date_str + ".log" << endl;
		outFile << "\n" << "Read Net : " << fsys::path(DetectModelPath) << "\n"
			<< "数据集 : " << fsys::path(ImgROOT) << "\n"
			<< "图像数量 : " << img_Filenames.size() << "\t\t 置信度 : " << Trust << "\n"
			<< "检测用时: " << times[2] / 60000 << "min " << times[2] % 60000 / 1000 << "s (total)  "
				<< times[0] << "ms (max)  " << times[1] << "ms (min)  "
				<< times[2] / img_Filenames.size() << "ms (average)" << "\n"
			<< "总目标数: " << res[3] << "  总扫描次数: " << img_Filenames.size() / set_num << "\n"
			<< "检出数量: " << res[0] << "  检出率: " << 100.0 * res[0] / res[3] << "%" << "\n"
			<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "\n"
			<< "误检次数: " << res[5] << "  误检率1: " << 100.0 * res[5] / (img_Filenames.size() / set_num) << "%" << "\n"
			<< "误检率3: " << 100.0 * res[2] / (res[0] + res[2]) << "%" << "\n"
			<< "违禁品总数: " << res[6] << "  误报违禁品数: " << res[7] << "  误检率2: " << 100.0 * res[7] / res[6] << "%" << endl;
		outFile.close();
	}
	std::cout << "图像数量 : " << img_Filenames.size() << "\n"
		<< "检测用时: " << times[2] / 60000 << "min " << times[2] % 60000 / 1000 << "s (total)  "
				<< times[0] << "ms (max)  " << times[1] << "ms (min)  "
				<< times[2] / img_Filenames.size() << "ms (average)" << "\n"
		<< "总目标数: " << res[3] << "  总扫描次数: " << img_Filenames.size() / set_num << "\n"
		<< "检出数量: " << res[0] << "  检出率: " << 100.0 * res[0] / res[3] << "%" << "\n"
		<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "\n"
		<< "误检次数: " << res[5] << "  误检率1: " << 100.0 * res[5] / (img_Filenames.size() / set_num) << "%" << "\n"
		<< "误检率3: " << 100.0 * res[2] / (res[0] + res[2]) << "%" << "\n"
		<< "违禁品总数: " << res[6] << "  误报违禁品数: " << res[7] << "  误检率2: " << 100.0 * res[7] / res[6] << "%" << endl;
	return 0;
}


// 多平面对抗集测试

// int main() {
// 	vector<string> dataset = { "24_4_1/多平面", "24_3_20", "24_3_5", "24_1_12", "24_1_4", "23_12_20" };
// 	string ImgROOT = "/media/data/dataset3/MPB测试集/" + dataset[data_n-1] + "/"; //数据集根目录
// 	//string ImgROOT = "/media/data/dataset3/MPB测试集/多平面民航测试库/戴手腕饰品（女）/";
// 	string outPath = workplace + "runs/detect/" + ModelName + "/"; //输出结果的保存位置
// 	fsys::create_directories(outPath);
// 	Yolo DPL(Trust);// 检测相关函数类实例化
// 	Net net;// 模型类实例化
// 	if (DPL.readModel(net, DetectModelPath)) {// 读模型
// 		std::cout << "Read Net : " << fsys::path(DetectModelPath) << "\n"
// 			<< "数据集 : " << fsys::path(ImgROOT) << "\n"
// 			<< "置信度 : " << Trust << endl;
// 	}
// 	else {
// 		std::cout << "read model fail!" << endl;
// 		return 0;
// 	}
// 	vector<int> res(8, 0);// 保存标记检出结果数目: 检出数、漏检数、误检数、总标记数、误检数备份、误检次数、总违禁品数、误检违禁品数
// 	vector<Output> DPLresult;// Output:Rect框、float置信度、int编号 使用前要clear
// 	vector<std::string> dir = {"."};
// 	for (int a = 0; a < dir.size(); a++) {
// 		res.assign(res.size(), 0);	//将res清零		
// 		string tmpPath = ImgROOT + "原始库/"; //缓存路径，图像所在目录
// 		cv::String img_Folder;
// 		// 数据库测试模式
// 		img_Folder = tmpPath + "*.*";
// 		std::vector<cv::String> img_Filenames;
// 		cv::glob(img_Folder, img_Filenames);
// 		std::sort(img_Filenames.begin(),img_Filenames.end());
// 		clock_t start = clock();
// 		for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {	
// 			tqdm(img_Count + 1, img_Filenames.size());
// 			// 读取当前样本对应的标记
// 			const string imgname = fsys::path(img_Filenames[img_Count]).filename().string();
// 			string prefix = imgname.substr(0, imgname.size() - 8);
// 			string markPath = ImgROOT + "标注库/" + imgname.substr(0, imgname.size() - 4) + ".xml";
// 			string markPath_LA = ImgROOT + "标注库/" + prefix + "_L_A.xml";
// 			string markPath_LB = ImgROOT + "标注库/" + prefix + "_L_B.xml";
// 			string markPath_RA = ImgROOT + "标注库/" + prefix + "_R_A.xml";
// 			string markPath_RB = ImgROOT + "标注库/" + prefix + "_R_B.xml";
// 			vector<cv::Rect> mark_LA, mark_LB, mark_RA, mark_RB, mark;
// 			if (!(DPL.ReadMark(markPath, mark) && DPL.ReadMark(markPath_LA, mark_LA) && DPL.ReadMark(markPath_LB, mark_LB)
// 				&& DPL.ReadMark(markPath_RA, mark_RA) && DPL.ReadMark(markPath_RB, mark_RB))) {
// 					std::cout << "标记读取错误" << endl;
// 					continue;
// 			}// 表示该样本没有对应的标记文件，跳到下一张
// 			Mat imgInitial = imread(img_Filenames[img_Count]);
// 			DPLresult.clear();
// 			DPL.Detect(imgInitial, net, DPLresult);// 检测
// 			// 检测框后处理: 根据最小框重叠面积整合重叠框
// 			DPL.mergeRect(DPLresult);
// 			vector<int> markCmp = CompareMark(DPLresult, mark, res);
// 			if ((img_Count + 1) % 4 == 0) { // 一次扫描检测完成
//                 // 解析违禁品个数
//                 int j = imgname.size() - 9, count = 0;
// 			    while (j > 25 && !isalpha(imgname[j])) {
//                     if (imgname[--j] == '_') {
//                         count++;
//                     }
//                 }
//                 res[6] += count / 3;
// 			    if (res[4] < res[2]) {	// 出现误检			
// 				    res[5]++;	// 更新误检次数
// 				    res[4] = res[2];
//                     res[7] += count / 3;	// 更新误检违禁品数	
// 			    }
// 			}
// 			if (DetecMOD == 0)
// 				continue;// 不输出可视化结果，当前图像任务已结束
// 			// 调试模式: 在原图上框出检测结果，在imagewatch上显示	
// 			Mat imgOri_LA = imread(tmpPath + prefix + "_L_A.png"), imgOri_LB = imread(tmpPath + prefix + "_L_B.png"),
// 				imgOri_RA = imread(tmpPath + prefix + "_R_A.png"), imgOri_RB = imread(tmpPath + prefix + "_R_B.png");
// 			Mat imgInitial_LA = imgOri_LA.clone(), imgInitial_LB = imgOri_LB.clone(),
// 				imgInitial_RA = imgOri_RA.clone(), imgInitial_RB = imgOri_RB.clone();
// 			for (const auto& i : mark) {
// 				cv::rectangle(imgInitial, i, Scalar(255, 0, 0), 3);
// 			}
// 			for (const auto& i : mark_LA) {
// 				cv::rectangle(imgInitial_LA, i, Scalar(255, 0, 0), 3);
// 			}
// 			for (const auto& i : mark_LB) {
// 				cv::rectangle(imgInitial_LB, i, Scalar(255, 0, 0), 3);
// 			}
// 			for (const auto& i : mark_RA) {
// 				cv::rectangle(imgInitial_RA, i, Scalar(255, 0, 0), 3);
// 			}
// 			for (const auto& i : mark_RB) {
// 				cv::rectangle(imgInitial_RB, i, Scalar(255, 0, 0), 3);
// 			}
// 			for (auto& i : DPLresult) {
// 				// string cofi = to_string(int(round(i.confidence * 10000)));
// 				// cofi.insert(cofi.length() - 2, "."); // 插入小数点
// 				// cv::putText(imgInitial, cofi, Point(i.box.x - 15, i.box.y - 15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 0.3);
// 				cv::rectangle(imgInitial, i.box, Scalar(0,0,255), 2);
// 			}
// 			Mat imgComb(imgInitial.rows, imgInitial.cols * 5, CV_8UC3);
// 			imgInitial.copyTo(imgComb(Rect(0, 0, imgInitial.cols, imgInitial.rows)));
// 			imgInitial_LA.copyTo(imgComb(Rect(imgInitial.cols, 0, imgInitial.cols, imgInitial.rows)));
// 			imgInitial_RA.copyTo(imgComb(Rect(imgInitial.cols * 2, 0, imgInitial.cols, imgInitial.rows)));
// 			imgInitial_LB.copyTo(imgComb(Rect(imgInitial.cols * 3, 0, imgInitial.cols, imgInitial.rows)));
// 			imgInitial_RB.copyTo(imgComb(Rect(imgInitial.cols * 4, 0, imgInitial.cols, imgInitial.rows)));
// 			if (DetecMOD == 1) {		//仅调试，不保存结果
// 				continue;
// 			}
// 			// 保存结果
// 			else if (DetecMOD == 2) {
// 				vector<string> outRecFilePath = {};
// 				// if (markCmp[0] == 0 && markCmp[1] == 0) {	//既没漏检，也没误检
// 				// 	outRecFilePath.emplace_back("rightRec");
// 				// }
// 				//			
// 				// if (markCmp[0] == 1) {	//出现漏检
// 				// 	outRecFilePath.emplace_back("missRec");
// 				// 	string suffix = imgname.substr(imgname.size() - 8, 4);
// 				// 	Mat imgRelational, imgRelationalOri;
// 				// 	if ( suffix == "_L_A") {
// 				// 		imgRelational = imgInitial_RA;
// 				// 		imgRelationalOri = imgOri_RA;						
// 				// 	}	
// 				// 	else if (suffix == "_R_A") {
// 				// 		imgRelational = imgInitial_LA;
// 				// 		imgRelationalOri = imgOri_LA;				
// 				// 	}
// 				// 	else if (suffix == "_L_B") {
// 				// 		imgRelational = imgInitial_RB;
// 				// 		imgRelationalOri = imgOri_RB;
// 				// 	}
// 				// 	else {
// 				// 		imgRelational = imgInitial_LB;
// 				// 		imgRelationalOri = imgOri_LB;
// 				// 	}	
// 				// 	DPLresult.clear();
// 				// 	DPL.Detect(imgRelationalOri, net, DPLresult);
// 				// 	DPL.mergeRect(DPLresult);
// 				// 	for (auto& i : DPLresult) {
// 				// 		// string cofi = to_string(int(round(i.confidence * 10000)));
// 				// 		// cofi.insert(cofi.length() - 2, "."); // 插入小数点
// 				// 		// cv::putText(imgRelational, cofi, Point(i.box.x - 15, i.box.y - 15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 0.3);
// 				// 		cv::rectangle(imgRelational, i.box, Scalar(0, 0, 255), 2);
// 				// 	}
// 				// 	imgComb = Mat::zeros(imgInitial.rows, imgInitial.cols * 4, CV_8UC3);
// 				// 	imgInitial.copyTo(imgComb(Rect(0, 0, imgInitial.cols, imgInitial.rows)));
// 				// 	imgRelational.copyTo(imgComb(Rect(imgInitial.cols, 0, imgInitial.cols, imgInitial.rows)));
// 				// 	if (suffix.substr(suffix.size() - 1) == "A") {
// 				// 		imgInitial_LB.copyTo(imgComb(Rect(imgInitial.cols * 2, 0, imgInitial.cols, imgInitial.rows)));
// 				// 		imgInitial_RB.copyTo(imgComb(Rect(imgInitial.cols * 3, 0, imgInitial.cols, imgInitial.rows)));
// 				// 	}
// 				// 	else {
// 				// 		imgInitial_LA.copyTo(imgComb(Rect(imgInitial.cols * 2, 0, imgInitial.cols, imgInitial.rows)));
// 				// 		imgInitial_RA.copyTo(imgComb(Rect(imgInitial.cols * 3, 0, imgInitial.cols, imgInitial.rows)));
// 				// 	}
// 				// }				
// 				if (markCmp[1] == 1){
// 					outRecFilePath.emplace_back("errorRec");
// 				}			
// 				if (!outRecFilePath.empty()) {
// 					string filename = imgname.substr(0, imgname.size() - 4) + ".jpg";
// 					for (string s : outRecFilePath) {
// 						string savePath = outPath + to_string(Trust).substr(0, 4) + "/" + s;
// 						fsys::create_directories(savePath);
// 						imwrite(savePath + "/" + filename, imgComb);
// 					}
// 				}					
// 			}
// 			else
// 				continue;
// 		}
// 		clock_t end = clock();
// 		std::time_t currentTime = std::time(nullptr);
// 		std::tm localtime;
// 		localtime_r(&currentTime, &localtime);
// 		string date_str = to_string(localtime.tm_mon + 1) + "_" + to_string(localtime.tm_mday);
// 		std::ofstream outFile(outPath + date_str + ".log", std::ios::app);	//以添加的方式打开
// 		if (!outFile.is_open())
// 			cout << "未能打开 : " << outPath + date_str + ".log" << endl;
// 		else {
// 			cout << "输出结果记录在 : " << outPath + date_str + ".log" << endl;
// 			outFile << "\n" << "Read Net : " << fsys::path(DetectModelPath) << "\n"
// 				<< "数据集 : " << fsys::path(ImgROOT) << "\n"
// 				<< "置信度 : " << Trust << "\n" 
// 				<< "图像数量 : " << img_Filenames.size() << "  扫描次数: " << img_Filenames.size() / 4 << "\n"				
// 				<< "单张图像检测时间: " << (double)(end - start) / (img_Filenames.size() * CLOCKS_PER_SEC) << "\n"
// 				<< "总目标数: " << res[3] << "  检出数量: " << res[0] << "  检出率: " << 100 * (double)(res[0]) / ((double)(res[3])) << "%" << "\n"
// 				<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "  误检率0: " << 100 * (double)(res[2]) / (double)(res[0] + res[2]) << "%" << "\n"
// 				<< "误检次数: " << res[5] << "  误检率: " << 100 * (double)(res[5]) / (double)(img_Filenames.size() / 4) << "%" << "\n"
// 				<< "违禁品总数: " << res[6] << "  误报违禁品数: " << res[7] << "  误检率2: " << 100 * (double)res[7] / (double)res[6] << "%" << endl;			 	
// 			outFile.close();
// 		}			
// 		std::cout << "\n" << "图像数量 : " << img_Filenames.size() << "  扫描次数: " << img_Filenames.size() / 4 << "\n"
// 			<< "单张图像检测时间: " << (double)(end - start) / (img_Filenames.size() * CLOCKS_PER_SEC) << "\n"
// 			<< "总目标数: " << res[3] << "  检出数量: " << res[0] << "  检出率: " << 100 * (double)(res[0]) / ((double)(res[3])) << "%" << "\n"
// 			<< "漏检数量: " << res[1] << "  误检数量: " << res[2] << "  误检率0: " << 100 * (double)(res[2]) / (double)(res[0] + res[2]) << "%" << "\n"
// 			<< "误检次数: " << res[5] << "  误检率: " << 100 * (double)(res[5]) / (double)(img_Filenames.size() / 4) << "%" << "\n"
// 			<< "违禁品总数: " << res[6] << "  误报违禁品数: " << res[7] << "  误检率2: " << 100 * (double)res[7] / (double)res[6] << "%" << endl;			 	
// 	}
// 	return 0;
// }


// 提取测试集

// int main() {
// 	string path1 = "/home/star/WorkSpace/Dataset/dataset2/5_29B/image_path/test.txt";
// 	string path2 = "/home/star/WorkSpace/Dataset/dataset3/5_29B/原始库/";
// 	string path3 = "/home/star/WorkSpace/Dataset/dataset3/5_29B/标注库/";
// 	fsys::create_directories(path2);
// 	fsys::create_directories(path3);
// 	ifstream file(path1);
// 	if (!file.is_open())
// 		return -1;
// 	vector<string> filepaths;
// 	filepaths.reserve(10000);
// 	string line;
// 	while (getline(file, line)) {
// 		filepaths.push_back(line);
// 	}
// 	int worker_nums = 8;
// 	cout << "Total Nums : " << filepaths.size() << "\nWorker Nums : " << worker_nums << endl;
// 	vector<int> set_size(worker_nums, filepaths.size() / worker_nums);
// 	vector<std::thread> threads(worker_nums);
// 	if (filepaths.size() % worker_nums) {
// 		set_size.emplace_back(filepaths.size() % worker_nums);
// 		threads.emplace_back(thread());
// 	}
// 	int finished = 0;
// 	std::chrono::time_point start = std::chrono::system_clock::now();
// 	for (int i = 0; i < threads.size(); i++) {
// 		int start_index = i * set_size[0];
// 		threads[i] = thread([path2, path3,i,start_index,  &filepaths, &set_size, &finished, &start]{
// 			for (int j = 0; j < set_size[i]; j++) {
// 				string current_filepath = filepaths[start_index + j];
// 				fsys::copy(current_filepath, path2, fsys::copy_options::overwrite_existing);
// 				current_filepath.replace(current_filepath.find("images"), 6, "Annotations");
// 				current_filepath.replace(current_filepath.size()-4, 4, ".xml");
// 				fsys::copy(current_filepath, path3, fsys::copy_options::overwrite_existing);
// 				finished++;
// 				tqdm(finished, filepaths.size(), start);
// 			}
// 		});
// 	}
// 	for (auto& thread : threads) {
// 		thread.join();
// 	}
// 	file.close();
// 	std::chrono::time_point end = std::chrono::system_clock::now();
// 	auto used_time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
// 	cout << "use time : " << used_time / 60 << ":" << used_time % 60 << endl;
// 	return 0;
// }


// 制作曲线

// int main() {
// 	//vector<string> dataset = { "24_6_29", "24_6_20", "24_4_22", "24_3_20", "24_3_5", "24_1_25", "24_1_12", "24_1_4", "23_12_20" };
// 	//string ImgROOT = "/home/star/WorkSpace/Dataset/dataset3/" + dataset[data_n-1] + "/"; //数据集根目录
// 	//string ImgROOT = "/home/star/WorkSpace/Dataset/dataset1/bmpimages/bmp_24_06_25/HWT多平面全回波训练库_40（共计4898张）20240624/训练库/";
// 	string ImgROOT = "/home/star/WorkSpace/Dataset/dataset3/50w_chouqu_A/";
// 	string outPath = workplace + "runs/detect/" + ModelName + "/"; //输出结果的保存位置
//  	fsys::create_directories(outPath);
// 	Yolo DPL(0.1);// 检测相关函数类实例化
// 	Net net;// 模型类实例化
// 	if (DPL.ReadModel(DetectModelPath)) {// 读模型
// 		std::cout << "Read Net : " << fsys::path(DetectModelPath) << "\n"
// 			<< "数据集 : " << fsys::path(ImgROOT) << "\n"
// 			<< "置信度 : " << 0.1 << endl;
// 	}
// 	else {
// 		std::cout << "read model fail!" << endl;
// 		return 0;
// 	}
// 	vector<std::string> dir = {"."};
// 	for (int a = 0; a < dir.size(); a++) {
// 		string tmpPath = ImgROOT + "原始库/"; //缓存路径，图像所在目录
// 		cv::String img_Folder;
// 		// 数据库测试模式
// 		img_Folder = tmpPath + "*.*";
// 		std::vector<cv::String> img_Filenames;
// 		cv::glob(img_Folder, img_Filenames);
// 		std::sort(img_Filenames.begin(),img_Filenames.end());
// 		vector<vector<int>> allres(80, {0, 0, 0, 0, 0, 0, 0, 0});
// 		//tmpPath = ImgROOT + dir[a] + "/imag/";
// 		auto start = std::chrono::system_clock::now();
// 		for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {	
// 			tqdm(img_Count + 1, img_Filenames.size(), start);
// 			// 读取当前样本对应的标记
// 			const string imgname = fsys::path(img_Filenames[img_Count]).filename().string();
// 			string markPath = ImgROOT + "标注库/" + imgname.substr(0, imgname.size() - 4) + ".xml";
// 			vector<cv::Rect> mark;
// 			if (!(DPL.ReadMark(markPath, mark))) {
// 					std::cout << "标记读取错误" << endl;
// 					continue;
// 			}// 表示该样本没有对应的标记文件，跳到下一张
// 			Mat imgOri = imread(img_Filenames[img_Count]);
// 			vector<Output> DPLresult;// Output:Rect框、float置信度、int编号
// 			DPL.Detect(imgOri, DPLresult);// 检测
// 			int j = imgname.size() - 9, count = 0;
// 			if ((img_Count + 1) % set_num == 0) {
// 				// 解析违禁品个数
// 		    	while (j > 25 && !isalpha(imgname[j])) {
//             		if (imgname[--j] == '_') {
//             		    count++;
//             		}
//             	}
// 				for (auto& res : allres) {
// 					res[6] += (count + 2) /3;
// 				}
// 			}
// 			for (int i = 0; i < 80; i++){
// 				float t = (float)(i + 10) / 100.0;	
// 				vector<Output> temp;
// 				for (auto& j : DPLresult){
// 					if (j.confidence > t){
// 						temp.emplace_back(j);
// 					}
// 				}
// 				// 检测框后处理：根据最小框重叠面积整合重叠框
// 				//DPL.mergeRect(temp);
// 				vector<bool> markCmp = CompareMark(temp, mark, allres[i]);
// 				if ((img_Count + 1) % 2 == 0) {		
// 					if(allres[i][2] - allres[i][4] > 0) {// 误检数有所增加
// 						allres[i][5]++;	// 误检次数加1
// 						allres[i][7] += (count + 2) / 3;	
// 						allres[i][4] = allres[i][2];
// 					}
// 				}
// 			}		
// 		}
// 		ofstream file(outPath + "result.txt");
// 		file << "Threshold\t检出数\t误检数\t总标记数\t检出率\t误检率3\t误检率1\t误检率2" << endl;
// 		for (int i = 0; i < 80; i++){
// 			file << to_string(0.10 + 0.01 * i).substr(0, 4) << "\t"
// 				<< allres[i][0] << "\t"     //检出数
// 				<< allres[i][2] << "\t"     //误检数
// 				<< allres[i][3] << "\t"     //总标记数
// 				<< 100.0 * allres[i][1] / allres[i][3] << "%\t"     //检出率
// 				<< 100.0 * allres[i][2] / (allres[i][0] + allres[i][2]) << "%\t"     //误检率3
// 				<< 100.0 * allres[i][5] / (img_Filenames.size() / set_num) << "%\t"     //误检率1
// 				<< 100.0 * allres[i][7] / allres[i][6] << "%" << endl;     //误检率2
// 		}
// 		file.close();
// 	}
// 	return 0;
// }



// 线程池示例

// class task
// {
// public:
//    void process()
//    {
//        //cout << "run........." << endl;//测试任务数量
//        int i = 0;
//        while (i < 10)
//        {
//            std::this_thread::sleep_for(std::chrono::seconds(2));
//            cout << "thread " << std::this_thread::get_id() << " finished  " << ++i << endl;
//        }
//    }
// };
// int main(void) {
//    threadPool<task> pool(6);//6个线程，vector
//    int n = 10;
//    while (n--)
//    {   
//        task* tt = new task();//使用智能指针
//        pool.append(tt);//不停的添加任务，任务是队列queue，因为只有固定的线程数
//        cout << "添加的任务数量：" << pool.tasks_queue.size() << endl;
//        std::this_thread::sleep_for(std::chrono::seconds(1));
//        delete tt;
//    }
//    cout << "添加完毕" << endl;
//    return 0;
// }

// void process(void* arg)
// {
//     //cout << "run........." << endl;//测试任务数量
//     int i = 0;
//     while (i < 10)
//     {
//         std::this_thread::sleep_for(std::chrono::seconds(2));
//         cout << "thread " << std::this_thread::get_id() << " finished  " << ++i << endl;
//     }
// }
// int main(void) {
//     ThreadPool pool(6);//6个线程，vector
//     int n = 50;
//     while (n--)
//     {   
// 		int a = 5;
//         pool.pushJob(process, &a, sizeof(int));//不停的添加任务，任务是队列queue，因为只有固定的线程数
//         //cout << "添加的任务数量：" << pool.tasks_queue.size() << endl;
//         std::this_thread::sleep_for(std::chrono::seconds(3));
//     }
//     cout << "添加完毕" << endl;
//     return 0;
// }


// 写入人体框

// int main() {
// 	string ImgROOT = "/home/star/WorkSpace/Dataset/dataset2/PoseDataset/train/bmpimages/HWT关键点训练库_66（正常姿态库8500张）20241223/";
// 	YoloPose DPLPose;
// 	Net net;// 模型类实例化
// 	if (DPLPose.readModel(net, PoseModelPath)) {// 读模型
// 		std::cout << "Read Net : " << fsys::path(PoseModelPath) << "\n"
// 			<< "数据集 : " << fsys::path(ImgROOT).parent_path() << endl;
// 	}
// 	else {
// 		std::cout << "\033[1;31mread model fail!\033[0m" << endl;
// 		return 0;
// 	}
// 	vector<int> res(3,0);//保存检出结果数目: 检出数、误检数、总数
// 	vector<string> dir = { "." };
// 	for (int a = 0; a < dir.size(); ++a) {
// 		res.assign(res.size(), 0);	//将res清零
// 		string tmpPath = ImgROOT + "原始库/";//缓存路径，图像文件夹所在目录
// 		cv::String img_Folder;		
// 		// 数据库测试模式
// 		img_Folder = tmpPath + "*.*";
// 		std::vector<cv::String> img_Filenames;
// 		cv::glob(img_Folder, img_Filenames);
// 		std::sort(img_Filenames.begin(), img_Filenames.end());
// 		auto start = std::chrono::system_clock::now();
// 		for (int img_Count = 0; img_Count < img_Filenames.size(); img_Count++) {	
// 			tqdm(img_Count + 1, img_Filenames.size(), start);
// 			// 读取当前样本对应的标记
// 			string imgname = fsys::path(img_Filenames[img_Count]).filename().string();
// 			string markPath = ImgROOT + "标注库/" + imgname.substr(0, imgname.size() - 4) + ".xml";
// 			Mat imgInitial = imread(img_Filenames[img_Count]);
// 			PoseOutput PoseResult;
// 			DPLPose.PoseDetect(imgInitial, net, PoseResult);// 姿态检测
// 			if (!DPLPose.WriteMark(markPath, PoseResult)) {
// 				cout << "写入失败：" << markPath << endl;
// 			}
// 		}
// 	}
// 	return 0;
// }


// CUDA 核函数，处理图像
// __global__ void processImageCUDA(unsigned char* image, int rows, int cols) {
//     // 获取线程在图像中的位置
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     // 确保线程在图像范围内
//     if (row < rows && col < cols) {
//         // 在这里执行图像处理操作，例如简单的像素值反转
//         image[row * cols + col] = 255 - image[row * cols + col];
//     }
// }
// int main() {
//     // 读取图像
//     cv::Mat image = cv::imread("/home/hanz/workspace/hanz/C++/yolov5/runs/detect/24_4_22/0.40/errorRec/2023-11-25-11-46-17_JY01_M_24.4_win_7_4_2_3_4_5_1_6_4_4_7_4_24_2_4_A.jpg", cv::IMREAD_GRAYSCALE);
//     if (image.empty()) {
//         std::cerr << "Failed to read image." << std::endl;
//         return -1;
//     }
//     // 获取图像大小
//     int rows = image.rows;
//     int cols = image.cols;
//     // 分配内存并将图像数据从主机复制到设备
//     size_t imageSize = rows * cols * sizeof(unsigned char);
//     unsigned char* d_image;
//     cudaMalloc(&d_image, imageSize);
//     cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);
//     // 定义CUDA网格和线程块大小
//     dim3 blockSize(16, 16);
//     dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
//     // 调用CUDA核函数处理图像
//     processImageCUDA<<<gridSize, blockSize>>>(d_image, rows, cols);
//     // 将处理后的图像数据从设备复制回主机
//     cudaMemcpy(image.data, d_image, imageSize, cudaMemcpyDeviceToHost);
//     // 释放设备内存
//     cudaFree(d_image);
//     // 显示处理后的图像
//     cv::imshow("Processed Image", image);
//     cv::waitKey(0);
//     return 0;
// }


// 锐化扩充训练集

// int main() {
//     string path1 = "/home/star/WorkSpace/Dataset/dataset2/PoseDataset/train/";
//     string savepath = "/home/star/WorkSpace/Dataset/dataset2/PoseDataset/train/";
//     fsys::create_directories(savepath + "images");
//     fsys::create_directories(savepath + "Annotations");
//     vector<String> imgfiles;
//     cv::glob(path1 + "images/*.*", imgfiles);
//     int n = imgfiles.size(), workernum = 8;
//     int set_num = (imgfiles.size() + workernum - 1) / workernum;
//     vector<float> args = { 0.4, 0.8, 1.2, -0.4, -0.8, -1.2 };
//     thread threads[workernum];
// 	int finished = 0;
//  	std::chrono::time_point start = std::chrono::system_clock::now();
//     for (int i = 0; i < workernum; i++) {
//         threads[i] = thread([set_num, i, n, &imgfiles, &args, &path1, &savepath, &finished, &start] {
//             int startind = i * set_num, endind = min((i + 1) * set_num, n);
//             for (int imgCount = startind; imgCount < endind; imgCount++) {
//                 string imgname = fsys::path(imgfiles[imgCount]).filename().string();
//                 string prefix = imgname.substr(0, imgname.size() - 4);
//                 string xmlpath = path1 + "Annotations/" + prefix + ".xml";
//                 Mat imgsrc = imread(imgfiles[imgCount], -1);
//                 Mat imgdst;
//                 for (auto& val : args) {
//                     cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
// 		                -val, -val, -val,
// 		                -val, 1 + 8 * val, -val,
// 		                -val, -val, -val);
//                     filter2D(imgsrc, imgdst, -1, kernel);
//                     string imgsavepath = savepath + "images/" + to_string(val).substr(0, 4) + "_"  + prefix + ".bmp";
//                     imwrite(imgsavepath, imgdst);
//                     string xmlsavepath = savepath + "Annotations/" + to_string(val).substr(0, 4) + "_"  + prefix + ".xml";
//                     fsys::copy(xmlpath, xmlsavepath);
//                 }
// 				finished++;
//  				tqdm(finished, imgfiles.size(), start);
//             }
//         });
//     }
//     for (int i = 0; i < workernum; i++) {
//         threads[i].join();
//     }
// 	std::chrono::time_point end = std::chrono::system_clock::now();
//  	auto used_time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
//  	cout << "use time : " << used_time / 60 << " : " << used_time % 60 << endl;
//     return 0;
// }



// int main() {
// 	string path = "/home/star/WorkSpace/Dataset/dataset2/PoseDataset/train/bmpimages/bmp_24_06_03/HWT多平面全回波训练库_30_姿态关键点（共计7226张）20240531/";
//     string savepath = "/home/star/WorkSpace/Dataset/dataset2/PoseDataset/train/bmpimages/bmp_24_06_03/modify/";
//     fsys::create_directories(savepath + "原始库");
//     fsys::create_directories(savepath + "标注库");
//     vector<string> filepaths;
//     cv::glob(path + "标注库/*.xml", filepaths);
//     YoloPose DPLPose;
//     auto start = std::chrono::system_clock::now();
//     for (int i = 0; i < filepaths.size(); i++) {
//         tqdm(i+1, filepaths.size(), start);
//         string filename = fsys::path(filepaths[i]).filename().string();
//         PoseOutput mark;
//         DPLPose.ReadMark(filepaths[i], mark);
//         if (mark.keypoints[7].visible == 0 || mark.keypoints[8].visible == 0) {
//             fsys::copy(filepaths[i], savepath + "标注库/" + filename, fsys::copy_options::skip_existing);
//             string imgpath = path + "原始库/" + filename.substr(0, filename.size() - 4) + ".bmp";
//             Mat img = imread(imgpath, 1);
//             cv::rectangle(img, mark.box, Scalar(255,0,0), 3);
// 			for (auto p : mark.keypoints) {
// 				cv::circle(img, p.point, 7, Scalar(255,0,0), cv::FILLED);
//                 string cofi = to_string(int(p.visible));
// 			    cv::putText(img, cofi, p.point - Point(15, 15), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 2, 0.3);
// 			}
//             imwrite(savepath + "原始库/" + filename.substr(0, filename.size() - 4) + ".jpg", img);
//         }
//     }
// 	return 0;
// }


// 将民航库添加到验证集

// int main() {
//     string valpath = "/home/star/WorkSpace/Dataset/dataset2/HWTMLB8-3/image_path/train.txt";
//     string path = "/home/star/WorkSpace/Dataset/dataset1/bmpimages/bmp_24_8_3/民航测试明显漏检标注图库/民航测试明显漏检标注图库/";
//     vector<string> files;
//     ofstream val(valpath, std::ios_base::app);
//     if (!val.is_open()) {
// 		cout << "文件未打开！" << endl;
// 		return -1;
// 	}
//     for (auto& iter : fsys::recursive_directory_iterator(path)) {
//         if (iter.path().extension().string() == ".png") {
//             for (int i = 0; i < 8; i++)
// 				val << iter.path().string() << endl;
//         }
//     }
//     return 0;
// }



