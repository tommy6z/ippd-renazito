//Paralelização do Problema K-means através de MPI
//Fábio A. de Siqueira

#define _POSIX_C_SOURCE 199309L  // Necessário para CLOCK_MONOTONIC
#include <limits.h>              // Para LLONG_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Header correto para clock_gettime e struct timespec
#include <mpi.h> // incluindo o MPI

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int* coords;     // Vetor de coordenadas inteiras
  int cluster_id;  // ID do cluster ao qual o ponto pertence
} Point;
//aaa
// --- Funções Utilitárias ---

/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com coordenadas inteiras.
 * Usa 'long long' para evitar overflow no cálculo da distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point* p1, Point* p2, int D) {
  long long dist = 0;
  for (int i = 0; i < D; i++) {
    long long diff = (long long)p1->coords[i] - p2->coords[i];
    dist += diff * diff;
  }
  return dist;
}

// --- Funções Principais do K-Means ---

/**
 * @brief Lê os dados de pontos (inteiros) de um arquivo de texto.
 */
void read_data_from_file(const char* filename, Point* points, int M, int D) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Erro: Não foi possível abrir o arquivo '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < D; j++) {
      if (fscanf(file, "%d", &points[i].coords[j]) != 1) {
        fprintf(stderr, "Erro: Arquivo de dados mal formatado ou incompleto.\n");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}

/**
 * @brief Inicializa os centroides escolhendo K pontos aleatórios do dataset.
 */
void initialize_centroids(Point* points, Point* centroids, int M, int K, int D) {
  srand(42);  // Semente fixa para reprodutibilidade

  int* indices = (int*)malloc(M * sizeof(int));
  for (int i = 0; i < M; i++) {
    indices[i] = i;
  }

  for (int i = 0; i < M; i++) {
    int j = rand() % M;
    int temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  for (int i = 0; i < K; i++) {
    memcpy(centroids[i].coords, points[indices[i]].coords, D * sizeof(int));
  }

  free(indices);
}

/**
 * @brief Fase de Atribuição: Associa cada ponto ao cluster do centroide mais próximo.
 */
void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D) {
  for (int i = 0; i < M; i++) {
    long long min_dist = LLONG_MAX;
    int best_cluster = -1;

    for (int j = 0; j < K; j++) {
      long long dist = euclidean_dist_sq(&points[i], &centroids[j], D);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = j;
      }
    }
    points[i].cluster_id = best_cluster;
  }
}

/**
 * @brief Fase de Atualização: Recalcula a posição de cada centroide como a média
 * (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */
void update_centroids(Point* points, Point* centroids, int M, int K, int D) {
  long long* cluster_sums = (long long*)calloc(K * D, sizeof(long long));
  int* cluster_counts = (int*)calloc(K, sizeof(int));

  for (int i = 0; i < M; i++) {
    int cluster_id = points[i].cluster_id;
    cluster_counts[cluster_id]++;
    for (int j = 0; j < D; j++) {
      cluster_sums[cluster_id * D + j] += points[i].coords[j];
    }
  }

  for (int i = 0; i < K; i++) {
    if (cluster_counts[i] > 0) {
      for (int j = 0; j < D; j++) {
        // Divisão inteira para manter os centroides em coordenadas discretas
        centroids[i].coords[j] = cluster_sums[i * D + j] / cluster_counts[i];
      }
    }
  }

  free(cluster_sums);
  free(cluster_counts);
}

/**
 * @brief Imprime os resultados finais e o checksum (como long long).
 */
void print_results(Point* centroids, int K, int D) {
  printf("--- Centroides Finais ---\n");
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    printf("Centroide %d: [", i);
    for (int j = 0; j < D; j++) {
      printf("%d", centroids[i].coords[j]);
      if (j < D - 1) printf(", ");
      checksum += centroids[i].coords[j];
    }
    printf("]\n");
  }
  printf("\n--- Checksum ---\n");
  printf("%lld\n", checksum);  // %lld para long long int
}

/**
 * @brief Calcula e imprime o tempo de execução e o checksum final.
 * A saída é formatada para ser facilmente lida por scripts:
 * Linha 1: Tempo de execução em segundos (double)
 * Linha 2: Checksum final (long long)
 */
void print_time_and_checksum(Point* centroids, int K, int D, double exec_time) {
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      checksum += centroids[i].coords[j];
    }
  }
  // Saída formatada para o avaliador
  printf("%lf\n", exec_time);
  printf("%lld\n", checksum);
}

// --- Função Principal ---
int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


  // Validação e leitura dos argumentos de linha de comando
  if (argc != 6) {
    fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* filename = argv[1];  // Nome do arquivo de dados
  const int M = atoi(argv[2]);     // Número de pontos
  const int D = atoi(argv[3]);     // Número de dimensões
  const int K = atoi(argv[4]);     // Número de clusters
  const int I = atoi(argv[5]);     // Número de iterações

  if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
    fprintf(stderr, "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
    return EXIT_FAILURE;
  }

  // --- Alocação de Memória ---
  int* all_coords = NULL;       // só rank 0 aloca para leitura do arquivo
  Point* points = NULL;         // só rank 0 usa essa estrutura completa
  int* centroids_coords = NULL; // centroides
  Point* centroids = NULL;

  if (rank == 0) 
  {
    // aloca pontos
    all_coords = (int*)malloc(M * D * sizeof(int));
    points = (Point*)malloc(M * sizeof(Point));
    for (int i = 0; i < M; i++) 
        points[i].coords = &all_coords[i * D];

    read_data_from_file(filename, points, M, D);

    // aloca e inicializa centroides diretamente
    centroids_coords = (int*)malloc(K * D * sizeof(int));
    centroids = (Point*)malloc(K * sizeof(Point));
    for (int i = 0; i < K; i++)
        centroids[i].coords = &centroids_coords[i * D];

    initialize_centroids(points, centroids, M, K, D);
}

  //calculando a divisão dos pontos entre processos
  int base = M/size;
  int resto = M % size;
  int local_M = base + (rank < resto ? 1 : 0);
  int* displs = malloc(size * sizeof(int));
  int* sendcounts = malloc(size * sizeof(int));
  int sum = 0;
  for (int i=0; i<size; i++) 
  {
    int m_i = base + (i < resto ? 1 : 0);
    sendcounts[i] = m_i * D;
    displs[i] = sum;
    sum += m_i * D;
  }

  //buffers regionais dos pontos dum processo
  int* local_coords = malloc(local_M * D * sizeof(int));
  Point* local_points = malloc(local_M * sizeof(Point));
  for (int i = 0; i < local_M; i++)
  {
    local_points[i].coords = &local_coords[i * D];
  }

  //distribuindo pontos
  MPI_Scatterv(all_coords, sendcounts, displs, MPI_INT, local_coords, local_M * D, MPI_INT, 0, MPI_COMM_WORLD );

  // --- Centroides com memória local válida em TODOS os processos ---
  // já foram alocados no rank 0, os outros recebem a memória via Bcast
  if (rank != 0) 
  {
      centroids_coords = (int*)malloc(K * D * sizeof(int));
      centroids = (Point*)malloc(K * sizeof(Point));
      for (int i = 0; i < K; i++)
          centroids[i].coords = &centroids_coords[i * D];
  }

  //broadcast dos centroides iniciais (agora todos têm memória válida!)
  MPI_Bcast(centroids_coords, K * D, MPI_INT, 0, MPI_COMM_WORLD);
 
  //buffers para soma e contagem 
  long long *local_sums = (long long*)calloc(K * D, sizeof(long long));
  long long *global_sums = (long long*)calloc(K * D, sizeof(long long));
  int *local_counts = (int*)calloc(K, sizeof(int));
  int *global_counts = (int*)calloc(K, sizeof(int));
  if (!local_sums || !global_sums || !local_counts || !global_counts) 
  {
    fprintf(stderr, "Erro: calloc falhou para buffers de soma/contagem\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // sincroniza e mede tempo com MPI_Wtime 
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  // laço principal do K-Means
  for (int iter = 0; iter < I; iter++) 
  {
    //zerando parciais locais
    memset(local_sums, 0, sizeof(long long) * K * D);
    memset(local_counts, 0, sizeof(int) * K);

    //1.cada rank usa seus pontos locais 
    for (int i = 0; i < local_M; i++) 
    {
      long long min_dist = LLONG_MAX;
      int best_cluster = -1;
      for (int c = 0; c < K; c++) 
      {
        long long dist = euclidean_dist_sq(&local_points[i], &centroids[c], D);
        if (dist < min_dist) { min_dist = dist; best_cluster = c; }
      }
      local_points[i].cluster_id = best_cluster;
      local_counts[best_cluster]++;
      for (int j = 0; j < D; j++) 
      {
        local_sums[best_cluster * D + j] += local_points[i].coords[j];
      }
    }

    // agregando resultados globalmente
    MPI_Allreduce(local_sums, global_sums, K * D, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // todos os ranks recalculam os centróides na sua cópia local
    for (int c = 0; c < K; c++) 
    {
      if (global_counts[c] > 0) 
      {
        for (int j = 0; j < D; j++) 
        {
          centroids[c].coords[j] = (int)(global_sums[c * D + j] / global_counts[c]);
        }
      }
    }
  }

  //sincroniza e mede tempo final
  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  double time_taken = t1 - t0;

  // --- Apresentação dos Resultados ---
  if (rank == 0) 
  {
    print_time_and_checksum(centroids, K, D, time_taken);
  }

  // --- Limpeza ---
  free(sendcounts);
  free(displs);
  free(local_coords);
  free(local_points);
  free(local_sums);
  free(global_sums);
  free(local_counts);
  free(global_counts);
  free(all_coords);
  free(points);
  free(centroids_coords);
  free(centroids);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
