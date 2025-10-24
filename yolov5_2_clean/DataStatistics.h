#pragma once
#include <iostream>
#include "DPL.h"
#include "DPLPose.h"

using namespace std;
using namespace cv;


std::vector<bool> CompareMark(vector<Output>& DPLresult, vector<cv::Rect>& mark, vector<int>& data);
std::vector<int> ComparePoseMark(PoseOutput& PoseResult, PoseOutput& mark, vector<int>& data, float Threshold = 0.03);
int PoseEvaluate(PoseOutput& PoseResult, int iPanel);
void tqdm(int current, int total, std::chrono::_V2::system_clock::time_point& start_time);	// 进度条工具
void tqdm(int current, int total);


// #include <vector>
// #include <queue>
// #include <thread>
// #include <iostream>
// #include <stdexcept>
// #include <condition_variable>
// #include <memory> //unique_ptr
// #include <assert.h>
// const int MAX_THREADS = 1000; //最大线程数目
// template <typename T>
// class threadPool
// {
// public:
// 	threadPool(int number = 1);//默认开一个线程
// 	~threadPool();
// 	std::queue<T* > tasks_queue; //任务队列
// 	bool append(T* request);//往请求队列＜task_queue＞中添加任务<T >
// private:
// 	//工作线程需要运行的函数,不断的从任务队列中取出并执行
// 	static void* worker(void* arg);
// 	void run();
// private:
// 	std::vector<std::thread> work_threads; //工作线程
// 	std::mutex queue_mutex;
// 	std::condition_variable condition; //必须与unique_lock配合使用
// 	bool stop;
// };//end class
// //构造函数，创建线程
// template <typename T>
// threadPool<T>::threadPool(int number) : stop(false)
// {
// 	if (number <= 0 || number > MAX_THREADS)
// 		throw std::exception();
// 	for (int i = 0; i < number; i++)
// 	{
// 		std::cout << "created Thread num is : " << i << std::endl;
// 		work_threads.emplace_back(worker, this);
// 		//添加线程
// 		//直接在容器尾部创建这个元素，省去了拷贝或移动元素的过程。
// 	}
// }
// template <typename T>
// inline threadPool<T>::~threadPool()
// {
// 	//std::unique_lock<std::mutex> lock(queue_mutex);
// 	stop = true;
// 	condition.notify_all();
// 	for (auto& ww : work_threads)
// 		ww.join();//可以在析构函数中join
// 	cout << "未完成的任务数量：" << tasks_queue.size() << endl;
// }
// //添加任务
// template <typename T>
// bool threadPool<T>::append(T* request)
// {
// 	//操作工作队列时一定要加锁，因为他被所有线程共享
// 	queue_mutex.lock();//同一个类的锁
// 	tasks_queue.push(request);
// 	queue_mutex.unlock();
// 	stop = true;
// 	condition.notify_one(); //线程池添加进去了任务，自然要通知等待的线程
// 	return true;
// }//单个线程
// template <typename T>
// void* threadPool<T>::worker(void* arg)
// {
// 	threadPool* pool = (threadPool*)arg;
// 	pool->run();//线程运行
// 	return pool;
// }
// template <typename T>
// void threadPool<T>::run()
// {
// 	while (!stop)
// 	{
// 		std::unique_lock<std::mutex> lk(this->queue_mutex);
// 		/*　unique_lock() 出作用域会自动解锁　*/
// 		this->condition.wait(lk, [this]
// 		{
// 			return !this->tasks_queue.empty() || stop;
// 		});//如果任务为空，则wait，就停下来等待唤醒//需要有任务，才启动该线程，不然就休眠
// 		if (tasks_queue.empty())
// 			continue;
// 		T* request = tasks_queue.front();
// 		tasks_queue.pop();
// 		lk.unlock();
// 		if (request)//来任务了，开始执行
// 			request->process();
// 	}
// }



// class ThreadPool{
// private:
//     struct NWORKER{
//         pthread_t threadid;
//         bool terminate;
//         int isWorking;
//         ThreadPool *pool;
//     } *m_workers;
//     struct NJOB{
//         void (*func)(void *arg);     //任务函数
//         void *user_data;
//     };
// public:
//     //线程池初始化
//     //numWorkers:线程数量
//     ThreadPool(int numWorkers, int max_jobs = 10);
//     //销毁线程池
//     ~ThreadPool();
//     //面向用户的添加任务
//     template <typename F>
//     int pushJob(F f);
//     int pushJob(void (*func)(void *), void *arg, int len);
// private:
//     //向线程池中添加任务
//     bool _addJob(NJOB* job);
//     //回调函数
//     static void* _run(void *arg);
//     void _threadLoop(void *arg);
// private:
//     std::list<NJOB*> m_jobs_list;
//     int m_max_jobs;							//任务队列中的最大任务数
//     int m_sum_thread;						//worker总数
//     int m_free_thread;						//空闲worker数
//     pthread_cond_t m_jobs_cond;           //线程条件等待
//     pthread_mutex_t m_jobs_mutex;         //为任务加锁防止一个任务被两个线程执行等其他情况
// };



