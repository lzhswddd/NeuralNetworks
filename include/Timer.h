#pragma once
class Timer
{
public:
	Timer();
	~Timer();
	void Start();
	double End();
protected:
	void* start;
	void* end;
	void* fc;
};