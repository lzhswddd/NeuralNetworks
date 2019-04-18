#ifndef __NETTREE_H_
#define __NETTREE_H_

#include "alignMalloc.h"

namespace nn
{
	enum NodeDire
	{
		PARENT = 0,
		SIBLING = 1,
		CHILD = 2,
	};

	template<typename _Tp>
	class TreeNode
	{
	public:
		_Tp data;
		TreeNode *ptr[3];
	};
	template<typename _Tp>
	class NetTree
	{
	public:
		NetTree();
	private:
		TreeNode *head;
		TreeNode *ctrl_l; 
		TreeNode *ctrl_r;
	};

	template<typename _Tp>
	inline NetTree<_Tp>::NetTree()
		: head((TreeNode<_Tp>*)fastMalloc(sizeof(TreeNode<_Tp>))), ctrl_l(head), ctrl_r(head)
	{

	}
}
#endif //__NETTREE_H_