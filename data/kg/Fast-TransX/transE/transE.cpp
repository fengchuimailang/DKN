#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <iostream>
#include <sstream>

using namespace std;

const float pi = 3.141592653589793238462643383;

int transeThreads = 8; // 8线程
int transeTrainTimes = 1000; //训练次数
int nbatches = 10;
int dimension = 50;  //定义好所有向量的长度都是50
float transeAlpha = 0.001;
float margin = 1;

// 指定输入输出路径

string inPath = "../../";
string outPath = "../../";


int *lefHead, *rigHead; //头左 头右
int *lefTail, *rigTail; // 尾左 尾右

struct Triple {   //定义实体关系三元组
	int h, r, t;
};

Triple *trainHead, *trainTail, *trainList;   // 三个三元组的指针

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {   // 从头开始比较的函数
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {  // 从尾开始比较的函数
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

/*
	There are some math functions for the program initialization.
*/
unsigned long long *next_random;

unsigned long long randd(int id) {
    // TODO 不懂，应该是随机计算迭代 next_random 数组
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
    // 0到 x-1的余数
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
    // TODO 不确定 应该是返回 min 和 max 之间的一个值
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
    // x 在 均值为 miu 标准差为 sigma的 正态分布中的值
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
    // TODO 这函数干啥的完全不懂
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma)); //dScope 是 0 到 正态分布最大值之间的一个随机值
	} while (dScope > y);  // 这个值得大于y
	return x;
}

void norm(float * con) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii)); //指针偏移再取值
	x = sqrt(x); // 向量的模
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

/*
	Read triples from the training file.
*/

int relationTotal, entityTotal, tripleTotal;   // 积累实体、关系、三元组 数量
float *relationVec, *entityVec; // 关系向量、实体向量
float *relationVecDao, *entityVecDao; // TODO 关系向量 实体向量

void init() {

	FILE *fin; // 文件输入指针
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r"); // 读文件
	tmp = fscanf(fin, "%d", &relationTotal); //按照 %d 从文件输入流中读取 直到空格或换行
	fclose(fin);

	relationVec = (float *)calloc(relationTotal * dimension, sizeof(float)); // 内存中分配个数为 relationTotal * dimension，长度为 sizeof(float)的空间
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
		    // randn 参数 miu sigma min max
		    // 生成关系向量的初始值
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal); //读实体数量
	fclose(fin);

	entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
		    // 生成实体向量的初始值
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec+i*dimension);  // 向量归一化
	}

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");  //三元组的文件
	tmp = fscanf(fin, "%d", &tripleTotal); //三元组数量
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple)); //申请内存 tripleTotal 个长度为 sizeof(Triple)的空间
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	tripleTotal = 0; 循环的下标
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);
		// 一行包含头、关系、尾 三个int

		// 再复制两遍
		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;

		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;
		tripleTotal++; // 移动下标
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head()); // 从头开始从小到大排序
	sort(trainTail, trainTail + tripleTotal, cmp_tail()); // 从尾开始从小到大排序

	lefHead = (int *)calloc(entityTotal, sizeof(int));  // 申请内存 entityTotal * sizof(int)
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(int)*entityTotal); // memset 新申请的内存做初始化
	memset(rigTail, -1, sizeof(int)*entityTotal);

	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) { // trainTail 队列中 相邻两三元组 t 不同
		    // TODO 下边两个是什么意思
			rigTail[trainTail[i - 1].t] = i - 1; //尾右
			lefTail[trainTail[i].t] = i; // 尾左边
		}
		if (trainHead[i].h != trainHead[i - 1].h) { // 相邻头不想等
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	// TODO 没太看明白
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1; // 指定最后一个head
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1; // 指定对后一个tail

	relationVecDao = (float*)calloc(dimension * relationTotal, sizeof(float)); // 申请向量空间
	entityVecDao = (float*)calloc(dimension * entityTotal, sizeof(float)); // 申请向量空间
}

/*
	Training process of transE.
*/

int transeLen;
int transeBatch;
float res;

float calc_sum(int e1, int e2, int rel) {
	float sum=0;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimension;
	// 下面应该是求 向量差的l1范数
    for (int ii=0; ii < dimension; ii++) {
        // fabs 求浮点数的绝对值
        sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
    }
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
    // TODO 猜测 a、b 应该一个是正例、一个是负例
    // 求各个向量头的地址
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimension;
	int lastb1 = e1_b * dimension;
	int lastb2 = e2_b * dimension;
	int lastbr = rel_b * dimension;
	for (int ii=0; ii  < dimension; ii++) {
		float x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		// transeAlpha = 0.001
		if (x > 0)
			x = -transeAlpha;
		else
			x = transeAlpha;
		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;
		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = transeAlpha;
		else
			x = -transeAlpha;
		relationVec[lastbr + ii] -=  x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
    // a 应该是正例 b是负例
	float sum1 = calc_sum(e1_a, e2_a, rel_a);
	float sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2) { // 超过margin 就梯度下降
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

int corrupt_head(int id, int h, int r) {   //corrupt 是使损坏的意思
    // 这个函数 是用例制造负样本的
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1; //头左
	rig = rigHead[h];   //头右
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;  // （头+尾） / 2
		if (trainHead[mid].r >= r) rig = mid;
		else lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
    // 这个函数 是用例制造负样本的
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transetrainMode(void *con) {
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = transeBatch / transeThreads; k >= 0; k--) { 多线程
		int j;
		int i = rand_max(id, transeLen);
		int pr = 500;
		if (randd(id) % 1000 < pr) {   //TODO 应该是计算概率 头尾一半一半
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		}
		// 向量归一化
		norm(relationVec + dimension * trainList[i].r);
		norm(entityVec + dimension * trainList[i].h);
		norm(entityVec + dimension * trainList[i].t);
		norm(entityVec + dimension * j);
	}
	pthread_exit(NULL); // 线程通过调用 pthread_exit 终止
}

void* train_transe(void *con) {
	transeLen = tripleTotal;
	transeBatch = transeLen / nbatches;  // 10个 batch 确定每个batch大小
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long)); // 申请 线程数 * sizeof(unsigned long long)
	for (int epoch = 0; epoch < transeTrainTimes; epoch++) { // 训练次数
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) { // batch数量
			pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t));
			// 多线程训练
			for (long a = 0; a < transeThreads; a++)
				pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
			for (long a = 0; a < transeThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transE.
*/

void out_transe() {
		stringstream ss;
		ss << dimension;
		string dim = ss.str();
	
		FILE* f2 = fopen((outPath + "TransE_relation2vec_" + dim + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "TransE_entity2vec_" + dim + ".vec").c_str(), "w");
		// 文件输出关系向量
		for (int i=0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		// 文档输出 实体向量
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
}

/*
	Main function
*/

int main() {
	time_t start = time(NULL);
	init();   // 初始化 读文件 申请内存
	train_transe(NULL); // 训练
	out_transe(); // 输出向量
	cout << time(NULL) - start << " s" << endl;
	return 0;
}
