#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 
#include <math.h> 
#include "som_structures.h"

char **strsplit(char *str_row, int n){
//this function splits str_row (the current read line from the textfile), split it (by ',' separator) into n char * and returns a pointer to the first char *
	int i = 0;
	char **tmp = (char **)malloc(n * sizeof(char *));
	char* token = NULL;
	const char sep[2]=",";
	token = strtok(str_row, sep);
	while (token != NULL)
    {
	        tmp[i] = token;
		token = strtok (NULL, sep);
		i++;
    }
	return tmp;
}

double normalize(double *vect,int n){
//this function normalize vect (of n doubles) and returns its norm
	int j;
	double norm=0.0;
	for (j=0;j<n;j++)
	       norm += pow(vect[j],2);
	norm = sqrt(norm);
	for (j=0;j<n;j++)
	       vect[j] = vect[j]/norm;
	return norm;
}

void print_dataset(Datanode *dataset, Params prm){
	int cri=0;
	while(cri<prm.n_rows){
		for(int j=0 ; j<prm.n_cols ; j++)
			printf("%f ",dataset[cri].vect[j]);
		printf("%s ",dataset[cri].label);
		printf("%f\n",dataset[cri].norm);
		
		cri++;
	}
}

//each j'th column of the returned vector is the average of all j'th components of the dataset 
double *avg_vector(Datanode *dataset,Params p){
	int i,j;
	double *res=(double *)malloc(p.n_cols*sizeof(double));
	for (j=0;j<p.n_cols;j++){
		double avg = 0.0;
		for (i=0;i<p.n_rows;i++){
			avg += dataset[i].vect[j]; 
			}	
		avg = avg/(double)p.n_rows;
		res[j]=avg;}
	return res;
}
//Neuron grid initianilization with vectors around the average vector of the input dataset
Node **init_grid(double *avg, Params p){
	Node **grid=(Node **)malloc(p.nb_r*sizeof(Node *)); 
	for(int i=0; i<p.nb_r; i++)
		grid[i]=(Node *)malloc(p.nb_c*sizeof(Node));
	//initialize the grid with random values 
	srand(time(NULL));
	for(int i=0; i<p.nb_r; i++){
		for(int j=0; j<p.nb_c; j++){
			Node current;
			current.w = (double *)malloc((p.n_cols)*sizeof(double));
			int k = 0;
			while(k<p.n_cols){
				int rd = rand()%(p.n_rows/2);//rd a random int in [0,n_row/2] (n_row number of datanodes of our input dataset
				current.w[k]=avg[k] + pow(-1,j)*(double)(rd/100.0); //if j%2==0 we add the random value else we substract it, here we'll have a uniform distribution around the average vector (note that i have devided rd by 100 because the values are normalized) 
//not that it is possible to have negative initial values for the grid
				k++;
				}
			grid[i][j] = current;
			}
		}
	return grid;
}
void print_grid(Node **grid, Params p){
	printf("-------\n");
	for(int i=0; i<p.nb_r; i++){
		for(int j=0; j<p.nb_c; j++){
			Node current = grid[i][j];
			int k = 0;
			while(k<p.n_cols){
				printf("%f ",current.w[k]);
				k++;
				}
			printf("\n");
		}
	}	
	printf("-------\n");
		
}

int *shuffle(int n){
/* Generate a Sequence of Unique Random Integers with values from 0 to n-1
(based on random permutations)
	Step 1 : we initialize an array of indexes from 1 to N*/
	int *rand_index=(int *)malloc(n*sizeof(int));
	for(int i=0; i<n; i++)
		rand_index[i] = i;
	//Step 2 : shuffle the created array
	int buff;
	int i, i0;
	srand(time(NULL));
	for(i=0; i<n; i++){
		i0=rand()%n;
		if(i0 != i)
		{
			buff = rand_index[i];
			rand_index[i] =rand_index[i0];
			rand_index[i0] = buff;
		}
	}
	//Step 3 we access the dataset elements with indexes from the shuffled array
	return rand_index;
}

//returns the ecleadean distance between vect1 and vect2, each of them has n_col components
double euclidean_dist(double * vect1, double * vect2, int n_cols){
	double dist = 0.0;
	for(int i=0; i<n_cols; i++)
		dist += pow(vect1[i]-vect2[i],2);
	return sqrt(dist);
}

BMU findBMU(double *vect, Node **grid, Params prm){
		BMU bmu_rd; //the best matching unit for the randomly selected datanode (which is dataset[rd])
		//we initialize bmu_rd components with the following values
		bmu_rd.l=0;
		bmu_rd.c=0;
		bmu_rd.act = RAND_MAX;

	for(int i=0; i<prm.nb_r; i++){
		for(int j=0; j<prm.nb_c; j++){
			double dist = euclidean_dist(grid[i][j].w, vect, prm.n_cols);
			if(dist<bmu_rd.act){
				bmu_rd.act = dist;
				bmu_rd.l = i;
				bmu_rd.c = j;
				}
 		}
	}//here we found out the BMU of the selected datanode
	return bmu_rd;
}


BMU_cell *add_bmu(BMU_cell *l_bmu, BMU bmu_rd){
// add a bmu to the stack of bmus
	BMU_cell *p_cell=(BMU_cell *)malloc(sizeof(BMU_cell));
	(p_cell->one_bmu).l = bmu_rd.l;
	(p_cell->one_bmu).c = bmu_rd.c;
	(p_cell->one_bmu).act = bmu_rd.act;
	p_cell->next=l_bmu;
	return p_cell;
}

void freeStack(BMU_cell *l_bmu){
	while(l_bmu!=NULL){	
		BMU_cell *ltmp=l_bmu;		
		l_bmu=l_bmu->next;
		free(ltmp); //we heve to free memory
		}
	free(l_bmu);
}

BMU_cell *otherBMUS(BMU bmu_rd, Node **grid, double *vect, Params p, int *n){
	BMU_cell *l_bmu=NULL;
	l_bmu=add_bmu(l_bmu, bmu_rd); //l_bmu --> [bmu_rd|next--]-->NULL
	for(int i=0; i<p.nb_r; i++){
		for(int j=0; j<p.nb_c; j++){
			double dist = euclidean_dist(grid[i][j].w, vect, p.n_cols);
			if(dist==bmu_rd.act && (i!=bmu_rd.l || j!=bmu_rd.c )){ // if it is another neuron which has the same activation value than bmu_rd
				(*n)++;
				BMU bmu2add;
				bmu2add.act = dist; bmu2add.l = i; bmu2add.c = j;
				l_bmu=add_bmu(l_bmu, bmu2add);
				}
 		}
	}
	return l_bmu;
}

BMU electBMU(BMU_cell *l_bmu, int nbmu){
//	srand(time(NULL));
	int rd = rand()%(nbmu);
	int i;
	BMU elected_bmu;
	for(i=0; i<rd; i++){
		BMU_cell *ltmp=l_bmu;		
		l_bmu=l_bmu->next;
		free(ltmp); //we heve to free memory
		}
	elected_bmu.l=(l_bmu->one_bmu).l; elected_bmu.c=(l_bmu->one_bmu).c; elected_bmu.act=(l_bmu->one_bmu).act;
//	printf("%d\n",nbmu);
	
//	printf("%d %d \n",(l_bmu->one_bmu).l, (l_bmu->one_bmu).c);
	return elected_bmu;
}

int *neighbors(BMU elected_bmu, int rad, Params p){
//returns an int array composed with the coordinates of the rectangle vertices which delimet the area of the elected_bmu's neighbors in the neurons grid
	int *rect= (int *)malloc(4*sizeof(int)); //neighbors coordinates delimited by : res[0]=the first row, res[1] the last row, res[2] the first column, res[3] the last column
	rect[0] = (elected_bmu.l-rad <= 0)? 0 : elected_bmu.l-rad;
	rect[1] = (elected_bmu.l+rad >= p.n_rows)? p.n_rows-1 : elected_bmu.l+rad;
	rect[2] = (elected_bmu.c-rad <= 0)? 0 : elected_bmu.c-rad;
	rect[3] = (elected_bmu.c+rad >= p.n_cols)? p.n_cols-1 : elected_bmu.c+rad;
	return rect;

}

void update_params(int nb_iter, Learn *p, int phase, Params prm,int total_iter){
	if(nb_iter == 0)
		{p->alpha = 0.7;
		p->rad = ceil(sqrt((prm.nb_r * prm.nb_c)/2)/2);}
	else if((nb_iter == phase * 1/3 || nb_iter == phase * 2/3) && p->rad > 1)
	    	p->rad = p->rad-1;
	else if(nb_iter == phase)
	    {p->rad = 2;
	    p->alpha *= 0.1;}
	else
	    {p->alpha *= (1.0f - (double) nb_iter/total_iter);}
}

int main(int argc, char **argv)
{

	FILE *fp;
	fp = fopen("iris.data","r");
	if (fp==NULL){
		puts("file not found or permission denied");
		exit(1);
	}
/*---------------------------------------------------	
STEP 1 : loading data in a dataset from input file
---------------------------------------------------*/
	char str_row[100]; //row input read from iris.data (it has char * type)
	char carriage_return;
	Params prm;
	//n_row an n_col are defined in the first line of the input file (separated by a space) 
	fscanf(fp, "%d", &(prm.n_rows));
	fscanf(fp, "%d", &(prm.n_cols));
	fscanf(fp, "%c", &carriage_return);//we have to get the carriage return for the first line to avoid errors
	// we allocate memory for ou dataset (we'll have p.n_rows Datanodes that we'll retrieve from the file input(iris.data) and transform and then import it to the dataset
	Datanode *dataset = (Datanode *)malloc(prm.n_rows * sizeof(Datanode));
	int cri=0; // current row index (counter to load read lines to the dataset) 
	while(fgets(str_row, 100, fp) != NULL){ 
		str_row[strlen(str_row)-1]='\0';
		//then we split the current row into an array of char * using strtoke function (see strsplit function defined above)
		//example "5.1,3.5,1.4,0.2,Iris-setosa" becomes ["5.1","3.5","1.4","0.2","Iris-setosa"]
		char **splitted_row = strsplit(str_row,prm.n_cols+1); //+1 for the label component 

		//first we allocate memory for the current index of the dataset
		dataset[cri].vect=(double *)malloc((prm.n_cols)*sizeof(double));
		dataset[cri].label=(char*)malloc(30*sizeof(char));
	
		char *foo;
		for(int j=0;j<prm.n_cols;j++) 
			dataset[cri].vect[j] = strtod(splitted_row[j], &foo);
		//from ["5.1","3.5","1.4","0.2"] we get [5.1000,3.5000,1.4000,0.2000] that we load to dataset[cri].vect

		//we normalize the vector of the current node before adding it to the dataset
		dataset[cri].norm = normalize(dataset[cri].vect,prm.n_cols);
		strcpy(dataset[cri].label, splitted_row[prm.n_cols]) ;//because splitted_row[n_col] contains the label of the input line
		free(splitted_row);
		cri++;
		}
//print_dataset(dataset, prm); //to print the dataset
prm.nb_r = 5; //number of neuron-grid rows (parameter to define)
prm.nb_c = 60;//number of neuron-grid columns (parameter to define)

/*---------------------------------------------------	
     STEP 2 : initialize neurons grid (init_grid)
---------------------------------------------------*/
//calculate the avg vector
double *avg = avg_vector(dataset, prm);
Node **grid = init_grid(avg, prm);
//print_grid(grid, prm); // to print the initial grid

//****Learning phase*****
	int total_iter = 500*prm.n_rows*0.25; //int total_iter = 500*prm.n_rows;
	int phase = total_iter * 0.25;
	Learn p;
/*----------------------------------------------------------	
     STEP 3 : select randomly a datanode from the dataset
-----------------------------------------------------------*/
	int *shuffled = shuffle(prm.n_rows);
	int i_datanode =-1; //indexe of the selected datanode (from the input dataset) 
	for(int iter=0; iter<total_iter; iter++){
		i_datanode = (i_datanode<prm.n_rows-1)? i_datanode+1 : 0;
		int rd = shuffled[i_datanode];//rd random int in [0, prm.n_rows-1]
/*---------------------------------------------------------------------------------------	
     STEP 4 : find out the best matching unit from the grid of the selected datanode
---------------------------------------------------------------------------------------*/	
		BMU bmu_rd = findBMU(dataset[rd].vect,grid, prm); //the best matching unit for the randomly selected datanode (which is dataset[rd])
	//then we'll see if there is more than one BMU for the same datanod, in this case we'll create a linked list to save the founded BMUs from which we'll select randomly only one BMU 
		int nbmu=1; //we have at least 1 bmu which is bmu_rd
		BMU_cell *l_bmu=otherBMUS(bmu_rd, grid, dataset[rd].vect, prm,&nbmu); //create stack of BMUs and modify the value of nbmu(number of bmus for the selected captor) we have to give the addresse of nbmu otherwise we can't modify it
		BMU elected_bmu = electBMU(l_bmu, nbmu);	
		freeStack(l_bmu);
/*---------------------------------------------------------
     STEP 5 : update the learning parameters
----------------------------------------------------------*/	
		update_params(iter, &p, phase, prm, total_iter);
		int *neighs=neighbors(elected_bmu, p.rad, prm);
		for(int i=neighs[0];i<=neighs[1];i++){
			for(int j=neighs[2];j<=neighs[3];j++){
				//update grid[i][j].w
				for(int k=0; k<prm.n_cols; k++)
					{grid[i][j].w[k] += p.alpha * (dataset[rd].vect[k] - grid[i][j].w[k]);}				
			}
		}
	double rapport=(double) iter/total_iter;
//	printf("\nProgression : %f%% alpha: %lf", rapport*100 ,p.alpha);	
	}
print_grid(grid, prm);	
	
	fclose(fp);
	return 0;
 }
