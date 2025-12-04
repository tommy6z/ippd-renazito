#define _POSIX_C_SOURCE 199309L  // Necessário para CLOCK_MONOTONIC
#include <limits.h>              // Para LLONG_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Header correto para clock_gettime e struct timespec
#include <pthread.h>
#include <unistd.h>

long thread_count;

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
// oi Niveaa
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

 typedef struct {
    long thread_id;
    int M, K, D, thread_count;
    Point *points;
    Point *centroids;
} ThreadArgs;

 void *points_to_cluster_thread(void *arg){
    ThreadArgs *data = (ThreadArgs*) arg;

    long my_id = data->thread_id;
    int M = data->M;
    int K = data->K;
    int D = data->D;
    int thread_count = data->thread_count;
    Point *points = data->points;
    Point *centroids = data->centroids;

    int local_M = M / thread_count;
    int start = my_id * local_M;
    int end = (my_id == thread_count - 1) ? M : start + local_M;
    
    for (int i = start; i < end; i++) {
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
    return NULL;
 }

void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D) {
    pthread_t *thread_handles;
    thread_handles = malloc(thread_count*sizeof(pthread_t));
    ThreadArgs *thread_args = malloc(thread_count * sizeof(ThreadArgs));
    for (long thread = 0; thread < thread_count; thread++) {
        thread_args[thread].thread_id = thread;
        thread_args[thread].M = M;
        thread_args[thread].K = K;
        thread_args[thread].D = D;
        thread_args[thread].thread_count = thread_count;
        thread_args[thread].points = points;
        thread_args[thread].centroids = centroids;

        pthread_create(&thread_handles[thread], NULL, points_to_cluster_thread, &thread_args[thread]);
    }

    for (int thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);
    free(thread_args);
}

/**
 * @brief Fase de Atualização: Recalcula a posição de cada centroide como a média
 * (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */

typedef struct {
    long thread_id;
    int M, K, D, thread_count;
    Point *points;
    long long *global_sums; 
    int *global_counts;     
    pthread_mutex_t *lock;
} UpdateArgs;

void *update_centroids_thread(void *arg) {
    UpdateArgs *a = (UpdateArgs*) arg;
    long id = a->thread_id;
    int local_M = a->M / a->thread_count;
    int start = id * local_M;
    int end = (id == a->thread_count - 1) ? a->M : start + local_M;

    long long *local_sums = calloc(a->K * a->D, sizeof(long long));
    int *local_counts = calloc(a->K, sizeof(int));

    for (int i = start; i < end; i++) {
        int c = a->points[i].cluster_id;
        if (c < 0 || c >= a->K) continue;
        local_counts[c]++;
        for (int j = 0; j < a->D; j++) {
            local_sums[c * a->D + j] += a->points[i].coords[j];
        }
    }

    pthread_mutex_lock(a->lock);
    for (int c = 0; c < a->K; c++) {
        a->global_counts[c] += local_counts[c];
        for (int j = 0; j < a->D; j++) {
            a->global_sums[c * a->D + j] += local_sums[c * a->D + j];
        }
    }
    pthread_mutex_unlock(a->lock);

    free(local_sums);
    free(local_counts);
    return NULL;
}

void update_centroids(Point* points, Point* centroids, int M, int K, int D) {
  long long* cluster_sums = (long long*)calloc(K * D, sizeof(long long));
  int* cluster_counts = (int*)calloc(K, sizeof(int));

pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    UpdateArgs *args = malloc(thread_count * sizeof(UpdateArgs));

    for (long t = 0; t < thread_count; t++) {
        args[t].thread_id = t;
        args[t].M = M;
        args[t].K = K;
        args[t].D = D;
        args[t].thread_count = thread_count;
        args[t].points = points;
        args[t].global_sums = cluster_sums;
        args[t].global_counts = cluster_counts;
        args[t].lock = &lock;
        pthread_create(&threads[t], NULL, update_centroids_thread, &args[t]);
    }
    for (int t = 0; t < thread_count; t++) pthread_join(threads[t], NULL);
    pthread_mutex_destroy(&lock);

    for (int c = 0; c < K; c++) {
        if (cluster_counts[c] > 0) {
            for (int j = 0; j < D; j++) {
                centroids[c].coords[j] = (int)(cluster_sums[c * D + j] / cluster_counts[c]);
            }
        }
    }
  free(threads);
  free(args);
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
  // Validação e leitura dos argumentos de linha de comando
  if (argc < 6) {
    fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes> [threads]\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* filename = argv[1];  // Nome do arquivo de dados
  const int M = atoi(argv[2]);     // Número de pontos
  const int D = atoi(argv[3]);     // Número de dimensões
  const int K = atoi(argv[4]);     // Número de clusters
  const int I = atoi(argv[5]);     // Número de iterações

  if (argc >= 7) {
        thread_count = atol(argv[6]);
    } else {
        thread_count = sysconf(_SC_NPROCESSORS_ONLN); // número de cores disponíveis
    }

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