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
//运行路径
string show_path()
{
	char buffer[MAX_PATH];
	_getcwd(buffer, MAX_PATH);
	return string(buffer);
}


/**
@brief getFiles 得到路径下所有文件的路径
@param path 文件夹路径
@param files 保存path下的所有文件路径
*/

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	intptr_t hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
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
