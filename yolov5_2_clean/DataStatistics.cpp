#include "DataStatistics.h"

std::mutex mtx;	// 互斥锁,用于多线程

std::vector<bool> CompareMark(vector<Output>& DPLresult, vector<cv::Rect>& mark, vector<int>& data) {
	int right = 0, miss = 0, error = 0;
	for (auto& i : mark) {
		bool flag = false;
		for (auto& j : DPLresult) {
			Rect inter = i & j.box;
			float uion = i.width * i.height + j.box.width * j.box.height - inter.width * inter.height;
			float iou = inter.width * inter.height / uion;
			if (iou > 0.1) {
				flag = true;
				break;
			}
		}
		if (flag == true) ++right;// 存在交集并跳出了，记录检出
		else ++miss;// 对于该标记，检出结果与其没有交集，记录漏检
	}
	for (auto& j : DPLresult) {
		bool flag = false;
		for (auto& i : mark) {
			Rect inter = i & j.box;
			float uion = i.width * i.height + j.box.width * j.box.height - inter.width * inter.height;
			float iou = inter.width * inter.height / uion;
			if (iou > 0.1) {
				flag = true;
				break;
			}
		}
		if (flag == false) ++error;// 检出结果与目标没有交集，记录误检
	}
	mtx.lock();
	data[0] += right;
	data[1] += miss;
	data[2] += error;
	data[3] += mark.size();
	mtx.unlock();
	std::vector<bool> result(2, false);
	if (miss > 0)
		result[0] = true;// 出现漏检
	if (error > 0)
		result[1] = true;// 出现误检
	return result;		//用于将检出结果分类保存
}

vector<int> ComparePoseMark(PoseOutput& PoseResult, PoseOutput& mark, vector<int>& data, float Threshold) {
	vector<int> error;
	float db = std::sqrt(pow(PoseResult.box.width, 2) + pow(PoseResult.box.height, 2)); //人体框对角线长度 
	for (int i = 0; i < mark.keypoints.size(); i++) {
		float ax = mark.keypoints[i].point.x, ay = mark.keypoints[i].point.y, av = mark.keypoints[i].visible;
		float bx = PoseResult.keypoints[i].point.x, by = PoseResult.keypoints[i].point.y, bv = PoseResult.keypoints[i].visible;
		if (ax == 0 && ay == 0) {
			if (abs(av / 2 - bv) > 0.5)
				error.push_back(i);
			continue;
		}
		float dp = std::sqrt(pow(ax - bx, 2) + pow(ay - by, 2));
		float k = dp / db;
		if (k > Threshold || abs(av / 2 - bv) > 0.5)
			error.push_back(i);
	}
	data[0] += mark.keypoints.size() - error.size();
	data[1] += error.size();
	data[2] += mark.keypoints.size();
	return error;
}

void tqdm(int current, int total, std::chrono::_V2::system_clock::time_point& start_time) {
	const int barWidth = 60;
	float fraction = static_cast<float>(current) / total;
	int numBars = static_cast<int>(fraction * barWidth);
	std::lock_guard<std::mutex> lock(mtx);	// 加锁，防止在多线程调用的情况下输出混乱
	std::cout << "  进度 : " << current << " / " << total << " [";
	for (int i = 0; i < numBars; ++i)
		if (i == numBars - 1)
			std::cout << ">";
		else
			std::cout << "=";
	for (int i = numBars; i < barWidth; ++i)
		std::cout << " ";
	auto current_time = std::chrono::system_clock::now();
	auto used_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
	std::cout << "] " << static_cast<int>(fraction * 100.0) << "%  "
		<< std::setw(2) << setfill('0') << used_time.count() / 60 << ":"
		<< std::setw(2) << setfill('0') << used_time.count() % 60 <<"\r";
	std::cout.flush();
	if (current == total)
		std::cout << std::endl;
}

void tqdm(int current, int total) {
	const int barWidth = 60;
	float fraction = static_cast<float>(current) / total;
	int numBars = static_cast<int>(fraction * barWidth);
	std::lock_guard<std::mutex> lock(mtx);	// 加锁，防止在多线程调用的情况下输出混乱
	std::cout << "  进度 : " << current << " / " << total << " [";
	for (int i = 0; i < numBars; ++i)
		if (i == numBars - 1)
			std::cout << ">";
		else
			std::cout << "=";
	for (int i = numBars; i < barWidth; ++i)
		std::cout << " ";
	std::cout << "] " << static_cast<int>(fraction * 100.0) << "%  \r";
	std::cout.flush();
	if (current == total)
		std::cout << std::endl;
}


// ThreadPool::ThreadPool(int numWorkers, int max_jobs) : m_sum_thread(numWorkers), m_free_thread(numWorkers), m_max_jobs(max_jobs){   //numWorkers:线程数量
//     if (numWorkers < 1 || max_jobs < 1){
//         perror("workers num error");
//     }
//     //初始化jobs_cond
//     if (pthread_cond_init(&m_jobs_cond, NULL) != 0)
//         perror("init m_jobs_cond fail\n");
//     //初始化jobs_mutex
//     if (pthread_mutex_init(&m_jobs_mutex, NULL) != 0)
//         perror("init m_jobs_mutex fail\n");
//     //初始化workers
//     m_workers = new NWORKER[numWorkers];
//     if (!m_workers){
//         perror("create workers failed!\n");
//     }
// 	//初始化每个worker
//     for (int i = 0; i < numWorkers; ++i){
//         m_workers[i].pool = this;
//         int ret = pthread_create(&(m_workers[i].threadid), NULL, _run, &m_workers[i]);
//         if (ret){
//             delete[] m_workers;
//             perror("create worker fail\n");
//         }
//         if (pthread_detach(m_workers[i].threadid)){
//             delete[] m_workers;
//             perror("detach worder fail\n");
//         }
//         m_workers[i].terminate = 0;
//     }
// }
// ThreadPool::~ThreadPool(){
// 	//terminate值置1
//     for (int i = 0; i < m_sum_thread; i++){
//         m_workers[i].terminate = 1;
//     }
//     //广播唤醒所有线程
//     pthread_mutex_lock(&m_jobs_mutex);
//     pthread_cond_broadcast(&m_jobs_cond);
//     pthread_mutex_unlock(&m_jobs_mutex);
//     delete[] m_workers;
// }
// //面向用户的添加任务
// int ThreadPool::pushJob(void (*func)(void *), void *arg, int len) {
//     struct NJOB *job = (struct NJOB*)malloc(sizeof(struct NJOB));
//     if (job == NULL){
//         perror("malloc");
//         return -2;
//     }
//     memset(job, 0, sizeof(struct NJOB));
//     job->user_data = malloc(len);
//     memcpy(job->user_data, arg, len);
//     job->func = func;
//     _addJob(job);
//     return 1;
// }
// bool ThreadPool::_addJob(NJOB *job) {
// 	//尝试获取锁
//     pthread_mutex_lock(&m_jobs_mutex);
//     //判断队列是否超过任务数量上限
//     if (m_jobs_list.size() >= m_max_jobs){
//         pthread_mutex_unlock(&m_jobs_mutex);
//         return false;
//     }
//     //向任务队列添加job
//     m_jobs_list.push_back(job);
//     //唤醒休眠的线程
//     pthread_cond_signal(&m_jobs_cond);
//     //释放锁
//     pthread_mutex_unlock(&m_jobs_mutex);
// 	return true;
// }
// //run为static函数
// void* ThreadPool::_run(void *arg) {
//     NWORKER *worker = (NWORKER *)arg;
//     worker->pool->_threadLoop(arg);
// }
// void ThreadPool::_threadLoop(void *arg) {
//     NWORKER *worker = (NWORKER*)arg;
//     while (1){
//         //线程只有两个状态：执行\等待
//         //查看任务队列前先获取锁
//         pthread_mutex_lock(&m_jobs_mutex);
//         //当前没有任务
//         while (m_jobs_list.size() == 0) {
//         	//检查worker是否需要结束生命
//             if (worker->terminate) break;
//             //条件等待直到被唤醒
//             pthread_cond_wait(&m_jobs_cond,&m_jobs_mutex);
//         }
//         //检查worker是否需要结束生命
//         if (worker->terminate){
//             pthread_mutex_unlock(&m_jobs_mutex);
//             break;
//         }
//         //获取到job后将该job从任务队列移出，免得其他worker过来重复做这个任务
//         struct NJOB *job = m_jobs_list.front();
//         m_jobs_list.pop_front();
// 		//对任务队列的操作结束，释放锁
//         pthread_mutex_unlock(&m_jobs_mutex);
//         m_free_thread--;
//         worker->isWorking = true;
//         //执行job中的func
//         job->func(job->user_data);
//         worker->isWorking = false;
//         free(job->user_data);
//         free(job);
//     }
//     free(worker);
//     pthread_exit(NULL);
// }

