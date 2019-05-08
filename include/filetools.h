#ifndef __FILETOOLS_H__
#define __FILETOOLS_H__

#include <io.h>
#include <direct.h>  
#include <stdio.h> 
#include <string>  
#include <vector> 
using std::vector;
using std::string;

static string createpath(string path, string dir)
{
	path = path + "\\" + dir + "\\";
	return path;
}

static string createfile(string filename)
{
	return filename.substr(filename.rfind('\\') + 1);
}

static string createtype(string filename)
{
	return filename.substr(filename.rfind('.') + 1);
}
//����·��
static string show_path()
{
	char buffer[260];
	_getcwd(buffer, 260);
	return string(buffer);
}


/**
@brief getFiles �õ�·���������ļ���·��
@param path �ļ���·��
@param files ����path�µ������ļ�·��
*/

static void getFiles(string path, vector<string>& files)
{
	//�ļ����  
	intptr_t hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

static vector<string> strsplit(string &str, char ch)
{
	size_t idx = 0;
	size_t offset = 0;
	vector<string> spl;
	while (true) {
		offset = str.find(ch, idx);
		if (offset == string::npos)break;
		spl.push_back(str.substr(idx, offset - idx));
		idx = offset + 1;
	}
	str = str.substr(idx);
	if (str != "")
		spl.push_back(str);
	return spl;
}

static vector<float> str2float(vector<string> &str)
{
	vector<float> v;
	for (string &s : str) 
		v.push_back(stof(s));
	return v;
}

#endif //__FILETOOLS_H__
