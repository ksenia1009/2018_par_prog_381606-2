#include <mpi.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <ctime>

std::default_random_engine generator(time(0));
std::uniform_int_distribution <int> dist(0, 10);

void CreateMatrix(double* matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for(int j = i; j < size; j++)
		{
			if(i == j)
				matrix[i * size + j] = dist(generator);
			else
				matrix[j * size + i] = matrix[i * size + j] =  dist(generator);
		}
	}
}
void CreateMatrix_example(double* matrix, int size)
{
	matrix[0] = 3;
	matrix[1] = 4;
	matrix[2] = 0;
	matrix[3] = 4;
	matrix[4] = -3;
	matrix[5] = 0;
	matrix[6] = 0;
	matrix[7] = 0;
	matrix[8] = 5;
}
void CreateVector(double* vector, int size)
{
	for (int i = 0; i < size; i++)
	{
		vector[i] = dist(generator);
	}
}
void CreateVector_example(double* vector, int size)
{
	vector[0] = 1;
	vector[1] = 5;
	vector[2] = 9;
}
void Print_SLU(double* matrix, double* vector, int Size)
{
	for (int i = 0; i < Size; i++)
	{
		for (int j = 0; j < Size; j++)
		{
			std::cout << matrix[Size * i + j] << " ";
		}
		std::cout << "\t";
		std::cout << " | " << vector[i] << "\n";
	}
}
void MatrixVectorComposition(double* rows, double* vector, double* result, int size, int rowNum) {
	for (int i = 0; i < rowNum; i++) {
		result[i] = 0;
		for (int j = 0; j < size; j++)
			result[i] += rows[i * size + j] * vector[j];
	}
}
void VectorVectorDifference(double* vec1, double* vec2, double* result, int size, int rowNum) {
	for (int i = 0; i < rowNum; i++) {
		result[i] = vec1[i] - vec2[i];
	}
}
void VectorVectorSum(double* vec1, double* vec2, double* result, int size, int rowNum) {
	for (int i = 0; i < rowNum; i++) {
		result[i] = vec1[i] + vec2[i];
	}
}
void Scalar(double* vec1, double* vec2, double& result, int size, int rowNum) {
	double res = 0;
	for (int i = 0; i < rowNum; i++) {
		res += vec1[i] * vec2[i];
	}
	MPI_Allreduce(&res, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
void DoubleVectorComposition(double num, double*& vec2, double*& result, int size, int rowNum) {
	for (int i = 0; i < rowNum; i++) {
		result[i] = num * vec2[i];
	}
}
void DataDistribution(double*& matrix, double*& rows, double*& vector, double*& result, double*& direct, double*& grad, int*& sendNum, int*& sendInd, int size, int rowNum, int ProcRank, int ProcNum) {
	int restRows = size; // Оставшиеся строки
	MPI_Bcast(vector, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	sendInd = new int [ProcNum];
	sendNum = new int [ProcNum];
	rowNum = (size / ProcNum);
	sendNum[0] = rowNum * size;
	sendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		restRows -= rowNum;
		rowNum = restRows / (ProcNum - i);
		sendNum[i] = rowNum * size;
		sendInd[i] = sendInd[i - 1] + sendNum[i - 1];
	}
	// Рассылаем строки
	MPI_Scatterv(matrix, sendNum, sendInd, MPI_DOUBLE, rows, sendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	result = new double [rowNum];
	direct = new double [rowNum];
	grad = new double [rowNum];
	for(int i = 0; i < rowNum; i++) {
		result[i] = 0;
		direct[i] = 0;
		grad[i] = - vector[sendInd[ProcRank] / size + i];
	}
}
void DataRecieve(int*& recNum, int*& recInd, int size, int ProcNum) {
	int restRows = size; // Строки, которые еще не распределены
	recNum = new int [ProcNum];
	recInd = new int [ProcNum];
	// Определяем расположение текущего "подвектора" в результирующем векторе
	recInd[0] = 0;
	recNum[0] = size / ProcNum;
	for (int i = 1; i < ProcNum; i++) {
		restRows -= recNum[i - 1];
		recNum[i] = restRows / (ProcNum - i);
		recInd[i] = recInd[i - 1] + recNum[i - 1];
	}
}
void SeqMatrixVectorComposition(double* matrix, double* vector, int size, double* res)
{
	for (int i = 0; i < size; i++)
	{
		res[i] = 0;
		for (int j = 0; j < size; j++)
		{
			res[i] += matrix[i * size + j] * vector[j];
		}
	}
}
double SeqScalar(double* vector1, double* vector2, int size)
{
	double res = 0;
	for (int i = 0; i < size; i++)
	{
		res += vector1[i] * vector2[i];
	}
	return res;
}
double* MethodSoprGradSeq(double* matrix, double* vector, int size, double eps)
{
	double* x; // Приближение
	double* gr; // Градиент
	double* Ax; // Произведение матрицы A и x
	double* d; // Вектор направления
	int numIter = 0; // Количество итераций
	x = new double[size];
	for(int i = 0; i < size; i++)
		x[i] = 0;
	gr = new double[size];
	Ax = new double[size];
	SeqMatrixVectorComposition(matrix, x, size, Ax);
	for(int i = 0; i < size; i++)
		gr[i] = vector[i] - Ax[i];
	d = new double[size];
	for(int i = 0; i < size; i++)
		d[i] = gr[i];
	double grSquare = SeqScalar(gr, gr, size);
	double* temp = new double[size];
	while (grSquare > eps)
	{
		numIter++;
		SeqMatrixVectorComposition(matrix, d, size, temp);
		double s = grSquare / SeqScalar(temp, d, size); // Смещение
		for(int i = 0; i < size; i++)
			x[i] = x[i] + s * d[i];

		double* grNew = new double[size];
		for(int i = 0; i < size; i++)
			grNew[i] = gr[i] - s * temp[i];
		double grNewSquare = SeqScalar(grNew, grNew, size);
		double beta = grNewSquare / grSquare;
		gr = grNew;
		grSquare = grNewSquare;
		for(int i = 0; i < size; i++)
			d[i] = gr[i] + beta * d[i];
	}

	return x;

	delete[] x;
	delete[] d;
	delete[] gr;
	delete[] temp;
	delete[] Ax;
}

int main (int argc, char **argv)
{
	int size = atoi(argv[1]); // Размер исходной матрицы и вектора
	double* matrix; // Исходная матрица
	double* vector; // Вектор правой части
	double* result; // Вектор-решение
	int* sendNum; // Количество отправленных процессу элементов
	int* sendInd; // Индекс первого среди них
	int* recNum; // Количество элементов, которые будет отправлять данный процесс
	int* recInd; // Индекс первого среди них
	double eps = atof(argv[2]); // Погрешность вычислений
	double* procRows; // Строки, выделенные данному процессу
	int rowNum; // Количество этих строк
	double* prevCurResult; // Предыдущее приближение
	double* curResult; // Текущее приближение
	double* prevDirect; // Предыдущий вектор направления
	double* direct; // Вектора направления
	double* prevGrad; // Предыдущий градиент
	double* grad;	// Градиент
	int ProcRank, ProcNum;
	int restRows; // Количество строк, которые еще не отправлены
	double* tmpVec;
	double s; // Смещение по выбранному направлению, вычисляемое на шаге 3 каждой итерации
	double start_time, end_time, start_time_seq, end_time_seq;
	double* resultSeq; // Результат последовательного алгоритма
	int iterCount = 0; // Количество итераций

	MPI_Init(&argc, &argv);	

	matrix = new double[size * size];
	vector = new double[size];
	result = new double[size];
	resultSeq = new double[size];

	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);                                  
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	if(ProcRank == 0)
	{
		CreateMatrix(matrix, size); // Вызов матрицы из файла или заданная система
		CreateVector(vector, size); // (симметричная и положительно определенная матрица)
		//CreateMatrix_example(matrix, size);
		//CreateVector_example(vector, size);
		std::cout << "System:" << std::endl;
		Print_SLU(matrix, vector, size);

		// Последовательный алгоритм
		start_time_seq = MPI_Wtime();
		resultSeq = MethodSoprGradSeq(matrix, vector, size, eps);
		end_time_seq = MPI_Wtime();
	}

	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// Расчет количества строк каждому процессу
	restRows = size;
	for(int i = 0; i < ProcRank; i++)
		restRows = restRows - restRows / (ProcNum - i);
	rowNum = restRows / (ProcNum - ProcRank);

	procRows = new double[rowNum * size];
	curResult = new double[rowNum];
	direct = new double[rowNum];
	grad = new double[rowNum];
	tmpVec = new double[rowNum];

	// Рассылка данных
	DataDistribution(matrix, procRows, vector, prevCurResult, prevDirect, prevGrad, sendNum, sendInd, size, rowNum, ProcRank, ProcNum);	
	DataRecieve(recNum, recInd, size, ProcNum);

	int flag = 0;
	double sum; //Суммарная погрешность
	double tmp1, tmp2;

	start_time = MPI_Wtime();

	// Метод сопряженных градиентов (параллельная версия)
	do
	{
		sum = 0;
		//-------------------- Шаг 1 --------------------//
		double* prevCurResultTmp = new double[size];
		MPI_Allgatherv(prevCurResult, recNum[ProcRank], MPI_DOUBLE, prevCurResultTmp, recNum, recInd, MPI_DOUBLE, MPI_COMM_WORLD);
		MatrixVectorComposition(procRows, prevCurResultTmp, tmpVec, size, rowNum); // Произведение матрицы на вектор приближение
		VectorVectorDifference(tmpVec, vector + sendInd[ProcRank] / size, grad, size, rowNum); // Разность произведения и вектора правой части
		//-------------------- Шаг 2 --------------------//
		double* parts = new double[2];
		Scalar(grad, grad, parts[0], size, rowNum); // Скалярное произведение текущих значений градиентов (1)
		Scalar(prevGrad, prevGrad, parts[1], size, rowNum); // Скалярное произведение предыдущих значений градиентов (2)
		DoubleVectorComposition((double)(parts[0] / parts[1]), prevDirect, tmpVec, size, rowNum); // Произведение частности (1) и (2) и предыдущего здачения вектора направления (3)
		VectorVectorDifference(tmpVec, grad, direct, size, rowNum); // Разность (3) и текущего значения градиента
		Scalar(grad, grad, tmp1, size, rowNum);
		//-------------------- Шаг 3 --------------------//
		double above; // Числитель дроби
		double* Dtmp = new double[size];
		Scalar(direct, grad, above, size, rowNum); // Скалярное произведение текущих значений градиента и вектора направления
		MPI_Allgatherv(direct, recNum[ProcRank], MPI_DOUBLE, Dtmp, recNum, recInd, MPI_DOUBLE, MPI_COMM_WORLD);
		MatrixVectorComposition(procRows, Dtmp, tmpVec, size, rowNum); // Произведение матрицы и текущего значения вектора направления (4)
		double beyond; // Знаменатель дроби
		Scalar(direct, tmpVec, beyond, size, rowNum); // Скалярное произведение текущего значения вектора направления и (4)
		s = -(double)(above / beyond); // Вычисление велечины смещения по выбранному направлению
		//-------------------- Шаг 4 --------------------//
		DoubleVectorComposition(s, direct, tmpVec, size, rowNum); // Произведение текущего значения вектора направления на смещение
		VectorVectorSum(tmpVec, prevCurResult, curResult, size, rowNum); // Вычисление нового приближения

		for(int i = 0; i < rowNum; i++)
			sum += fabs(curResult[i] - prevCurResult[i]);
		MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
		delete prevGrad; delete prevDirect; delete prevCurResult;
		delete[] prevCurResultTmp; delete[] parts; delete[] Dtmp;
		prevGrad = grad; 
		prevDirect = direct; 
		prevCurResult = curResult;
		grad = new double[rowNum]; 
		direct = new double[rowNum]; 
		curResult = new double[rowNum]; 
		flag++;
		iterCount++;
		Scalar(vector, vector, tmp2, size, rowNum);
	} while(/*(flag < size) && */((tmp1 / tmp2) > (eps * eps)));
	MPI_Gatherv(prevCurResult, rowNum, MPI_DOUBLE, result, recNum, recInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	end_time = MPI_Wtime();

	// Печать результата
	if(ProcRank == 0)
	{
		std::cout << "Size: " << size << std::endl;
		std::cout << "Eps: " << eps << std::endl;
		std::cout << "Method Sopr Grad result (parallel): " << std::endl;
		for(int i = 0; i < size; i++)
			std::cout << result[i] << " ";
		std::cout << std::endl;
		std::cout << "Time: " << end_time - start_time << " c" << std::endl;

		std::cout << "Method Sopr Grad result (sequential): " << std::endl;
		for(int i = 0; i < size; i++)
			std::cout << resultSeq[i] << " ";
		std::cout << std::endl;
		std::cout << "Time: " << end_time_seq - start_time_seq << " c" << std::endl;

		std::cout << "Iterations count: " << iterCount << std::endl;

		std::cout << "Porg: " << sum << std::endl;
	}

	MPI_Finalize();	

	delete[] matrix;
	delete[] vector;
	delete[] result;
	delete[] procRows;
	delete[] curResult;
	delete[] prevGrad;
	delete[] prevDirect;
	delete[] prevCurResult;
	delete[] grad;
	delete[] direct;

	return 0;
}
