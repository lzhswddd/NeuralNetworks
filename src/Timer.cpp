#include "Timer.h"
#if defined(__linux__)
#include <sys/time.h>
#include <unistd.h>
Timer::Timer()
{
	start = (struct timeval*)malloc(sizeof(struct timeval));
	end = (struct timeval*)malloc(sizeof(struct timeval));
}
Timer::~Timer()
{
	FREE_PTR(start);
	FREE_PTR(end);
}
void Timer::Start()
{
	gettimeofday((struct timeval*)start, NULL);
}
mat_t Timer::End()
{
	gettimeofday((struct timeval*)end, NULL);
	return _T(((struct timeval*)end->tv_sec - (struct timeval*)start->tv_sec) * 1000.0 + (struct timeval*)end->tv_usec - (struct timeval*)start->tv_usec);
}
static struct timeval t1, t2;
void lzh::tools::StartCounter()
{
	gettimeofday(&t1, NULL);
}
mat_t lzh::tools::EndCounter()
{
	gettimeofday(&t2, NULL);
	return _T((t2.tv_sec - t1.tv_sec) * 1000.0 + t2.tv_usec - t1.tv_usec);
}
void lzh::tools::Wait(uint ms)
{
	sleep(ms);
}
#elif defined(_WIN32)
#include <windows.h>  
#include <io.h>
#include <direct.h>  
Timer::Timer()
{
	start = new LARGE_INTEGER();
	end = new LARGE_INTEGER();
	fc = new LARGE_INTEGER();
	QueryPerformanceFrequency((LARGE_INTEGER*)fc);
}
Timer::~Timer()
{
	if (start != nullptr) { delete start; start = nullptr; }
	if (end != nullptr) { delete end; end = nullptr; }
	if (fc != nullptr) { delete fc; fc = nullptr; }
}
void Timer::Start()
{
	QueryPerformanceCounter((LARGE_INTEGER*)start);
}
double Timer::End()
{
	QueryPerformanceCounter((LARGE_INTEGER*)end);
	return double(((((LARGE_INTEGER*)end)->QuadPart - ((LARGE_INTEGER*)start)->QuadPart) * 1000.0) / ((LARGE_INTEGER*)fc)->QuadPart);
}
#endif