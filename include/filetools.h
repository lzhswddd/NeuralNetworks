#ifndef __FILETOOLS_H__
#define __FILETOOLS_H__

#include <io.h>
#include <direct.h>  
#include <stdio.h> 
#include <string>  
#include <vector> 
using std::vector;
using std::string;

string createpath(string path, string dir)
{
	path = path + "\\" + dir + "\\";
	return path;
}

string createfile(string filename)
{
	return filename.substr(filename.rfind('\\') + 1);
}

string createtype(string filename)
{
	return filename.substr(filename.rfind('.') + 1);
}
//����·��
string show_path()
{
	char buffer[MAX_PATH];
	_getcwd(buffer, MAX_PATH);
	return string(buffer);
}


/**
@brief getFiles �õ�·���������ļ���·��
@param path �ļ���·��
@param files ����path�µ������ļ�·��
*/

void getFiles(string path, vector<string>& files)
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

#endif //__FILETOOLS_H__
