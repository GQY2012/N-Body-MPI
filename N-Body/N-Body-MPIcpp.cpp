#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define DIM     2                       // ��άϵͳ
#define X       0                       // X ����
#define Y       1                       // Y ����
typedef double vect_t[DIM];             // ������������
const double G = 6.67e-11;              // ������������
int my_rank, comm_sz;                   // ���̱�ź��ܽ�����
MPI_Datatype vect_mpi_t;                // ʹ�õ�������������
vect_t *vel = NULL;                     // ȫ��С���ٶȣ����� 0 �Ž��̵����

void Usage(char* prog_name)// ����˵��
{
    fprintf(stderr, "usage: mpiexec -n <nProcesses> %s\n", prog_name);
    fprintf(stderr, "<nParticle> <nTimestep> <sizeTimestep> <outputFrequency> <g|i>\n");
    fprintf(stderr, "    'g': inite condition by random\n");
    fprintf(stderr, "    'i': inite condition from stdin\n");
    exit(0);
}

void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, double* delta_t_p, int* output_freq_p, char* g_i_p)// ��ȡ������Ϣ
{                                                                     // ���н��̾����øú�������Ϊ�м���ͨ�ţ���ֻ�� 0 �Ž��̴������
    if (my_rank == 0)
    {
        if (argc != 6)
            Usage(argv[0]);
        *n_p = strtol(argv[1], NULL, 10);
        *n_steps_p = strtol(argv[2], NULL, 10);
        *delta_t_p = strtod(argv[3], NULL);
        *output_freq_p = strtol(argv[4], NULL, 10);
        *g_i_p = argv[5][0];
        if (*n_p <= 0 || *n_p % comm_sz || *n_steps_p < 0 || *delta_t_p <= 0 || *g_i_p != 'g' && *g_i_p != 'i')// ����Ҫ����������
        {
            printf("illegal input\n");
            if (my_rank == 0)
                Usage(argv[0]);
            MPI_Finalize();
            exit(0);
        }
    }
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_steps_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(delta_t_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_freq_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g_i_p, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void Gen_init_cond(double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n)// �Զ����ɳ�ʼ���������н��̾����øú�������Ϊ�м���ͨ��
{                                                          
    const double mass = 10000, gap = 0.01, speed = 0;
    if (my_rank == 0)
    {
		int ny = ceil(sqrt(n));
		double y = 0.0;
        for (int i = 0; i < n; i++)
        {
            masses[i] = mass;
            pos[i][X] = i * gap;
			if ((i + 1) % ny == 0)
				++y;
            pos[i][Y] = y;
            vel[i][X] = 0.0;
            vel[i][Y] = 0.0;
        }
    }
    // ͬ��������λ����Ϣ���ַ��ٶ���Ϣ
    MPI_Bcast(masses, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pos, n, vect_mpi_t, 0, MPI_COMM_WORLD);
    MPI_Scatter(vel, loc_n, vect_mpi_t, loc_vel, loc_n, vect_mpi_t, 0, MPI_COMM_WORLD);
}

void Get_init_cond(double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n)// �ֹ������ʼ���������ƺ��� Gen_init_cond()
{
    if (my_rank == 0)
    {
        printf("For each particle, enter (in order): mass x-coord y-coord x-velocity y-velocity\n");
        for (int i = 0; i < n; i++)
        {
            scanf_s("%lf", &masses[i]);
            scanf_s("%lf", &pos[i][X]);
            scanf_s("%lf", &pos[i][Y]);
            scanf_s("%lf", &vel[i][X]);
            scanf_s("%lf", &vel[i][Y]);
        }
    }
    MPI_Bcast(masses, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pos, n, vect_mpi_t, 0, MPI_COMM_WORLD);
    MPI_Scatter(vel, loc_n, vect_mpi_t, loc_vel, loc_n, vect_mpi_t, 0, MPI_COMM_WORLD);
}

void Output_state(double time, double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n)// �����ǰ״̬
{
    MPI_Gather(loc_vel, loc_n, vect_mpi_t, vel, loc_n, vect_mpi_t, 0, MPI_COMM_WORLD);// �Ӹ����̾ۼ��ٶ���Ϣ�������
    if (my_rank == 0)
    {
        printf("Output_state, time = %.2f\n", time);
        for (int i = 0; i < n; i++)
			printf(" %3d X: %.5f Y: %.5f\n", i, pos[i][X], pos[i][Y]);
        printf("\n");
        fflush(stdout);
    }
}

void Compute_force(int loc_part, double masses[], vect_t loc_forces[], vect_t pos[], int n, int loc_n)// ����С�� part �ܵ�����������
{
    const int part = my_rank * loc_n + loc_part;
    int k;
    vect_t f_part_k;
    double len, fact;
    for (loc_forces[loc_part][X] = loc_forces[loc_part][Y] = 0.0, k = 0; k < n; k++)
    {
        if (k != part)
        {
            f_part_k[X] = pos[part][X] - pos[k][X];
            f_part_k[Y] = pos[part][Y] - pos[k][Y];
            len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
            fact = -G * masses[part] * masses[k] / (len * len * len);
            f_part_k[X] *= fact;
            f_part_k[Y] *= fact;
            loc_forces[loc_part][X] += f_part_k[X];
            loc_forces[loc_part][Y] += f_part_k[Y];           
        }
    }
}

void Update_part(int loc_part, double masses[], vect_t loc_forces[], vect_t loc_pos[], 
	vect_t loc_vel[], int n, int loc_n, double delta_t)// ����λ��
{
    const int part = my_rank * loc_n + loc_part;
    const double fact = delta_t / masses[part];  
    loc_pos[loc_part][X] += delta_t * loc_vel[loc_part][X];
    loc_pos[loc_part][Y] += delta_t * loc_vel[loc_part][Y];
    loc_vel[loc_part][X] += fact * loc_forces[loc_part][X];
    loc_vel[loc_part][Y] += fact * loc_forces[loc_part][Y];
}

int main(int argc, char* argv[])
{
    int n, loc_n, loc_part;     // С������ÿ����С��������ǰС��ѭ��������
    int n_steps, step;          // ����ʱ�䲽������ǰʱ��Ƭ��ѭ��������
    double delta_t;             // ����ʱ�䲽��
    int output_freq;            // �������Ƶ��    
    double *masses;             // С��������ÿ�����̶��У�һ����ʼ����ͬ���Ͳ��ٸı�
    vect_t *pos, *loc_pos;      // С��λ�ã�ÿ��ʱ��Ƭ������ɺ���Ҫͬ��
    vect_t *loc_vel;            // С���ٶȣ��ɸ����̷ֿ����棬�������ʱ����ͬ��
    vect_t *loc_forces;         // �����̵�С����������
    char g_i;                   // ��ʼ����ѡ�g Ϊ�Զ����ɣ�i Ϊ�ֹ�����
    double start, finish;       // ��ʱ��

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Type_contiguous(DIM, MPI_DOUBLE, &vect_mpi_t);// �ύ��Ҫ����������
    MPI_Type_commit(&vect_mpi_t);

    // ��ȡ��������ʼ������
    Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
    loc_n = n / comm_sz;                                    // Ҫ�� n % comm_sz == 0
    masses = (double*)malloc(n * sizeof(double));
    pos = (vect_t*)malloc(n * sizeof(vect_t));
    loc_pos = pos + my_rank * loc_n;
    loc_vel = (vect_t*)malloc(loc_n * sizeof(vect_t));
    loc_forces = (vect_t*)malloc(loc_n * sizeof(vect_t));
    if (my_rank == 0)
        vel = (vect_t*)malloc(n * sizeof(vect_t));
    if (g_i == 'g')
        Gen_init_cond(masses, pos, loc_vel, n, loc_n);
    else
        Get_init_cond(masses, pos, loc_vel, n, loc_n);

    // ��ʼ���㲢��ʱ
    if (my_rank == 0)
        start = MPI_Wtime();   
    for (step = 1; step <= n_steps; step++)
    {
        // ����ÿС������������С��״̬��Ȼ��ͬ��С��λ��
        for (loc_part = 0; loc_part < loc_n; Compute_force(loc_part++, masses, loc_forces, pos, n, loc_n));
        for (loc_part = 0; loc_part < loc_n; Update_part(loc_part++, masses, loc_forces, loc_pos, loc_vel, n, loc_n, delta_t));
        MPI_Allgather(MPI_IN_PLACE, loc_n, vect_mpi_t, pos, loc_n, vect_mpi_t, MPI_COMM_WORLD);
        if (step % output_freq == 0)
			Output_state(step * delta_t, masses, pos, loc_vel, n, loc_n);
    }
	// ��ӡ��ʱ
    if (my_rank == 0)
    {
        finish = MPI_Wtime();
        printf("Elapsed time = %f ms\n", (finish - start) * 1000);
        free(vel);
    }
    MPI_Type_free(&vect_mpi_t);
    free(masses);
    free(pos);
    free(loc_vel);
    free(loc_forces);
    MPI_Finalize();
    return 0;
}