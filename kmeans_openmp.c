#define _POSIX_C_SOURCE 199309L  // Necessário para CLOCK_MONOTONIC
#include <limits.h>              // Para LLONG_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Header correto para clock_gettime e struct timespec
#include <omp.h>

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int* coords;     // Vetor de coordenadas inteiras
  int cluster_id;  // ID do cluster ao qual o ponto pertence
} Point;
// --- Funções Utilitárias ---

/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com coordenadas inteiras.
 * Usa 'long long' para evitar overflow no cálculo da distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point* p1, Point* p2, int D) {
  long long dist = 0;
  #pragma omp simd reduction(+:dist)
  for (int i = 0; i < D; i++) {
    long long a = p1->coords[i];
    long long b = p2->coords[i];
    long long diff = a - b;
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
void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D) { //PARALELIZAR

  #pragma omp parallel for schedule(static)
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
void update_centroids(Point* points, Point* centroids, int M, int K, int D) { //PARALELIZAR
  long long* somaCoordenadasCluster = (long long*)calloc(K * D, sizeof(long long));
  int* pontosEmCadaCluster = (int*)calloc(K, sizeof(int));
 
  int num_threads = omp_get_max_threads();

  long long* somaLocal = (long long*)calloc(num_threads * K * D, sizeof(long long));
  int* contagemLocal = (int*)calloc(num_threads * K, sizeof(int));

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long* somas = &somaLocal[tid * K * D];
    int* contagens = &contagemLocal[tid * K];

    #pragma omp for schedule(static)
    for (int i = 0; i < M; i++) {
      int cluster_id = points[i].cluster_id;
      contagens[cluster_id]++;
      for (int j = 0; j < D; j++) {
        somas[cluster_id * D + j] += points[i].coords[j];
      }
    }
  }

  for (int t = 0; t < num_threads; t++)
  {
     for (int i = 0; i < K; i++) {
        pontosEmCadaCluster[i] += contagemLocal[t * K + i];
        for (int j = 0; j < D; j++) {
          somaCoordenadasCluster[i * D + j] += somaLocal[t * K * D + i * D + j];
        }
      }
  }

  free(somaLocal);
  free(contagemLocal);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < K; i++) {
    if (pontosEmCadaCluster[i] > 0) {
      for (int j = 0; j < D; j++) {
        // Divisão inteira para manter os centroides em coordenadas discretas
        centroids[i].coords[j] = somaCoordenadasCluster[i * D + j] / pontosEmCadaCluster[i];
      }
    }
  }

  free(somaCoordenadasCluster);
  free(pontosEmCadaCluster);
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
  int* all_coords = (int*)malloc((M + K) * D * sizeof(int));
  Point* points = (Point*)malloc(M * sizeof(Point));
  Point* centroids = (Point*)malloc(K * sizeof(Point));
  // ... (verificação de alocação) ...
  for (int i = 0; i < M; i++) {
    points[i].coords = &all_coords[i * D];
  }
  for (int i = 0; i < K; i++) {
    centroids[i].coords = &all_coords[(M + i) * D];
  }

  // --- Preparação (Fora da medição de tempo) ---
  read_data_from_file(filename, points, M, D);
  initialize_centroids(points, centroids, M, K, D);

  // --- Medição de Tempo do Algoritmo Principal ---
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);  // Inicia o cronômetro

  // Laço principal do K-Means (A única parte que será medida)
  for (int iter = 0; iter < I; iter++) {
    assign_points_to_clusters(points, centroids, M, K, D);
    update_centroids(points, centroids, M, K, D);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);  // Para o cronômetro

  // Calcula o tempo decorrido em segundos
  double time_taken = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

  // --- Apresentação dos Resultados ---
  print_time_and_checksum(centroids, K, D, time_taken);

  // --- Limpeza ---
  free(all_coords);
  free(points);
  free(centroids);

  return EXIT_SUCCESS;
}