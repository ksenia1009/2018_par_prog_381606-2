#include <mpi.h>
#include <random>
#include <iostream>
#include <iomanip>

std::default_random_engine generator(time(0));
std::uniform_int_distribution <int> dist(0, 10);
void CreateMatrix(int* matrix, int columns, int lines)
{
	for (int i = 0; i < columns * lines; i++)
	{
		matrix[i] = dist(generator);
	}
}
void CreateVector(int* vector, int count)
{
	for (int i = 0; i < count; i++)
	{
		vector[i] = dist(generator);
	}
}
void PrintMatrix(int* matrix, int columns, int lines)
{
    for (int i = 0; i < lines; i++) 
	{
        for (int j = 0; j < columns; j++) 
		{
            std::cout << matrix[columns * i + j] << "\t";
        }
        std::cout << std::endl;
    }
}
void PrintVector(int* vector, int count)
{
	for (int i = 0; i < count; i++)
	{
		std::cout << vector[i] << "\t";
	}
	std::cout << std::endl;
}
int* Scalar(int* matrix, int* vector, int lines, int columns)
{
	int* res = new int[lines];
	for (int i = 0; i < lines; i++)
	{
		res[i] = 0;
		for (int j = 0; j < columns; j++)
		{
			res[i] += matrix[i * columns + j] * vector[j];
		}
	}
	return res;
}
int main (int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int lines = atoi(argv[1]);
	int columns = atoi(argv[2]);	

	int ProcRank, ProcNum;

	MPI_Status Status;

	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);                                  
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	int* matrix = new int[lines * columns];
	int* vector = new int[columns];
	int* result = new int[lines];
	int* resultSeq = new int[lines];
	int* procRows;
	int* procResult;	
	int rowCount;	
	int* sendOffset = new int[ProcNum];
	int* sendCount = new int[ProcNum];
	int* recOffset = new int[ProcNum];
	int* recCount = new int[ProcNum];
	double start_time, end_time, start_time_seq, end_time_seq;
	int block = lines / ProcNum;
	int rows = lines;
	int lines_cpy1 = lines, lines_cpy2 = lines;

	// Инициализация и вывод исходных матрицы и вектора
	if(ProcRank == 0)
	{
		CreateMatrix(matrix, columns, lines);
		CreateVector(vector, columns);

		std::cout << "Matrix:" << std::endl;
		PrintMatrix(matrix, columns, lines);
		std::cout << "Vector:" << std::endl;
		PrintVector(vector, columns);

		// Последовательный алгоритм
		start_time_seq = MPI_Wtime();
		resultSeq = Scalar(matrix, vector, lines, columns);
		end_time_seq = MPI_Wtime();
	}

	start_time = MPI_Wtime();

	MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(vector, columns, MPI_INT, 0, MPI_COMM_WORLD);

	// Вычисление количества строк каждому процессу
	for (int i = 0; i < ProcRank; i++)
	{
		rows = rows - rows / (ProcNum - i);
	}
	rowCount = rows / (ProcNum - ProcRank); // Количество строк каждому процессу
	
	procRows = new int[rowCount * columns];
	procResult = new int[rowCount];

	// Распределение строк матрицы между процессами
	sendOffset[0] = 0;
	sendCount[0] = block * columns;
	for(int i = 1; i < ProcNum; i++)
	{
		lines_cpy1 -= block;
		block = lines_cpy1 / (ProcNum - i);
		sendCount[i] = block * columns;
		sendOffset[i] = sendOffset[i - 1] + sendCount[i - 1];
	}
	int count1 = sendCount[ProcRank];
	MPI_Scatterv(matrix, sendCount, sendOffset, MPI_INT, procRows, count1, MPI_INT, 0, MPI_COMM_WORLD);

	// Вычисление скалярного произведения
	procResult = Scalar(procRows, vector, rowCount, columns);

	// Сбор результата со всех процессов
	recCount[0] = lines / ProcNum;
	recOffset[0] = 0;
	for (int i = 1; i < ProcNum; i++)
	{
		lines_cpy2 -= recCount[i - 1];
		recCount[i] = lines_cpy2 / (ProcNum - i);
		recOffset[i] = recOffset[i - 1] + recCount[i - 1];
	}
	int count2 = recCount[ProcRank];
	MPI_Gatherv(procResult, count2, MPI_INT, result, recCount, recOffset, MPI_INT, 0, MPI_COMM_WORLD);

	end_time = MPI_Wtime();

	// Печать результата
	if(ProcRank == 0)
	{
		std::cout << "Result (parallel algorithm): " << std::endl;
        for (int i = 0; i < lines; i++) 
		{
            std::cout << result[i] << "\t";
        }
        std::cout << std::endl;
		
		std::cout << "Result (sequential algorithm): " << std::endl;
        for (int i = 0; i < lines; i++) 
		{
            std::cout << resultSeq[i] << "\t";
        }
        std::cout << std::endl;
		
	// Проверка идентичности результатов параллельного и последовательного алгоритмов
	int flag = 1;
	for (int i = 0; i < lines; i++)
		if (result[i] != resultSeq[i])
		{
			flag = 0;
			break;
		}
	if (flag == 1)
		std::cout << "Results are equal" << std::endl;
	else
		std::cout << "Results are different" << std::endl;
		
	std::cout << "Time (parallel algorithm): " << end_time - start_time << std::endl;
	std::cout << "Time (sequential algorithm): " << end_time_seq - start_time_seq << std::endl;
	}

	MPI_Finalize();	
	
	delete[] matrix;
	delete[] vector;
	delete[] result;
	delete[] sendCount;
	delete[] sendOffset;
	delete[] recCount;
	delete[] recOffset;
	delete[] procRows;
	delete[] procResult;
	delete[] resultSeq;

	return 0;
}
