#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <random>

std::default_random_engine generator(time(0));
std::uniform_int_distribution <int> dist(0, 100);

void CreateMatrix(int* matrix, int columns, int lines)
{
	for (int i = 0; i < columns * lines; i++)
	{
		matrix[i] = dist(generator);
	}
}

void PrintMatrix(int* matrix, int columns, int lines)
{
	for (int i = 0; i < lines; i++)
	{
		for(int j = 0; j < columns; j++)
		{
			std::cout <<matrix[columns * i + j] << "\t";
		}
		std::cout << std::endl;
	}

}

void FindMax(int* matr, int* arr_max, int columns, int lines)
{
	int temp_max = 0;
	for (int i = 0; i < columns; i++)	//on columns
	{
		for (int j = 0; j < lines; j++)	//on lines
		{
			if (matr[i * lines + j] > temp_max)
			{
				temp_max = matr[i * lines + j];
			}
		}
		arr_max[i] = temp_max;
		temp_max = 0;
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);    
	
	int lines = atoi(argv[1]);
	int columns = atoi(argv[2]);	

	int ProcRank, ProcNum;
	
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);                                  
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);                           
	
	int block = lines / ProcNum; 
	int remain = lines % ProcNum; 

	int* matrix = nullptr;	
	int* result = nullptr;
	int* sizes = nullptr;
	int* offset = nullptr; 
	int* max = new int[columns];

	if (ProcRank == 0) 
	{
		matrix = new int[lines * columns];

		CreateMatrix(matrix, columns, lines);
		PrintMatrix(matrix, columns, lines);

		sizes = new int[ProcNum];
		offset = new int[ProcNum];
		result = new int[columns];

		for(int i = 0; i < ProcNum; i++)
		{
			sizes[i] = block * columns;
			offset[i] = i * block * columns;
		}
		sizes[ProcNum - 1] = (block + remain) * columns;
		offset[ProcNum - 1] = lines * columns - (block + remain) * columns;
	}

	int recCount = block * columns;
	if (ProcRank == ProcNum - 1)
	{
		recCount = (block + remain) * columns;
	}
	int* recBuf = new int[recCount];
	MPI_Scatterv(matrix, sizes, offset, MPI_INT, recBuf, recCount, MPI_INT, 0, MPI_COMM_WORLD);

	for(int i = 0; i < recCount / columns; i++)
	{
		for(int j = 0; j < columns; j++)
		{
			max[j] = std::max(max[j], recBuf[i * columns + j]);
		}
	}

	MPI_Reduce(max, result, columns, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    	if (ProcRank == 0) 
	{
        std::cout << "Maximum element (parallel algorithm): " << std::endl;
        for (int i = 0; i < columns; i++) 
		{
            std::cout << result[i] << "\t";
        }
        std::cout << std::endl;
    	}
	
	MPI_Finalize();
	return 0;
}
